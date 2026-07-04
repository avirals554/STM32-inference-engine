/*
 * Portable inference engine for the 2-layer TinyTransformer trained in
 * ../../train_iteration2.py. No STM32 / HAL dependencies -- the same object
 * file links into both the STM32 firmware and the host verification harness.
 *
 * Memory budget (static .bss):
 *   x       : 32 KB   -- hidden state, persists across layers
 *   scratch : 32 KB   -- attention concat / FF scratch
 *   k_h     : 16 KB   -- per-head keys
 *   v_h     : 16 KB   -- per-head values
 *   ----------------
 *   total   : 96 KB   -- fits in the STM32F411's 128 KB SRAM
 *
 * Weight matrices live in flash as int8 with a per-tensor float scale.
 * Biases and LayerNorm gamma/beta are float32 for numerical stability.
 *
 * Weight layouts follow PyTorch's nn.Linear convention: W is [out, in] and
 * y = x @ W.T + b, so we compute y[o] = (sum_i x[i] * W[o,i]) * scale + b[o].
 */

#include "model.h"

#include <math.h>
#include <string.h>

#include "model_config.h"
#include "../generated/model_weights.h"

/* ----- static work buffers (96 KB, .bss) ------------------------------ */

static float x[CONTEXT_LENGTH * EMBED_DIM];
static float scratch[CONTEXT_LENGTH * EMBED_DIM];
static float k_h[CONTEXT_LENGTH * HEAD_DIM];
static float v_h[CONTEXT_LENGTH * HEAD_DIM];

/* ----- kernels -------------------------------------------------------- */

static inline float dot_x_int8(const float *xv, const int8_t *w, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        acc += xv[i] * (float)w[i];
    }
    return acc;
}

/* out[o] = (x . W[o]) * scale + bias[o], for o in [0, out_dim) */
static void linear(const float *xv, const int8_t *W, int in_dim, int out_dim,
                   float scale, const float *bias, float *out) {
    for (int o = 0; o < out_dim; o++) {
        float acc = dot_x_int8(xv, &W[o * in_dim], in_dim);
        out[o] = acc * scale + bias[o];
    }
}

static void layernorm_row(float *row, const float *gamma, const float *beta) {
    float sum = 0.0f;
    for (int d = 0; d < EMBED_DIM; d++) sum += row[d];
    float mean = sum / (float)EMBED_DIM;

    float var = 0.0f;
    for (int d = 0; d < EMBED_DIM; d++) {
        float diff = row[d] - mean;
        var += diff * diff;
    }
    var /= (float)EMBED_DIM;
    /* PyTorch's default eps = 1e-5 */
    float inv_std = 1.0f / sqrtf(var + 1e-5f);

    for (int d = 0; d < EMBED_DIM; d++) {
        row[d] = (row[d] - mean) * inv_std * gamma[d] + beta[d];
    }
}

/* Numerically stable softmax over first `len` entries, in place. */
static void softmax_prefix(float *a, int len) {
    float m = a[0];
    for (int i = 1; i < len; i++) if (a[i] > m) m = a[i];
    float s = 0.0f;
    for (int i = 0; i < len; i++) {
        a[i] = expf(a[i] - m);
        s += a[i];
    }
    float inv = 1.0f / s;
    for (int i = 0; i < len; i++) a[i] *= inv;
}

/* ----- embedding + layer helpers -------------------------------------- */

static void embed_tokens(const uint8_t tokens[CONTEXT_LENGTH]) {
    /* x[pos] = char_emb[tok] * s_char + pos_emb[pos] * s_pos */
    for (int pos = 0; pos < CONTEXT_LENGTH; pos++) {
        uint8_t tok = tokens[pos];
        const int8_t *ce = &w_char_embedding[tok][0];
        const int8_t *pe = &w_pos_embedding[pos][0];
        float *dst = &x[pos * EMBED_DIM];
        for (int d = 0; d < EMBED_DIM; d++) {
            dst[d] = (float)ce[d] * s_char_embedding
                   + (float)pe[d] * s_pos_embedding;
        }
    }
}

/*
 * Runs one full transformer layer in place on the global `x` buffer.
 * All matrix pointers are passed in so the same function serves both layers.
 */
static void run_layer(
    const int8_t *W_q, float s_q, const float *b_q,
    const int8_t *W_k, float s_k, const float *b_k,
    const int8_t *W_v, float s_v, const float *b_v,
    const int8_t *W_o, float s_o, const float *b_o,
    const int8_t *W_ff1, float s_ff1, const float *b_ff1,
    const int8_t *W_ff2, float s_ff2, const float *b_ff2,
    const float *ln1_gamma, const float *ln1_beta,
    const float *ln2_gamma, const float *ln2_beta)
{
    const float attn_scale = 1.0f / sqrtf((float)HEAD_DIM);

    /* --- attention: fill scratch[pos][d] = concat of head outputs ----- */
    for (int h = 0; h < NUM_HEADS; h++) {
        /* Precompute K_h and V_h for this head (float, 16 KB each). */
        for (int pos = 0; pos < CONTEXT_LENGTH; pos++) {
            const float *xv = &x[pos * EMBED_DIM];
            for (int d = 0; d < HEAD_DIM; d++) {
                int o = h * HEAD_DIM + d;
                k_h[pos * HEAD_DIM + d] =
                    dot_x_int8(xv, &W_k[o * EMBED_DIM], EMBED_DIM) * s_k + b_k[o];
                v_h[pos * HEAD_DIM + d] =
                    dot_x_int8(xv, &W_v[o * EMBED_DIM], EMBED_DIM) * s_v + b_v[o];
            }
        }

        /* For each query position i, compute Q_h[i], attention scores over
         * unmasked keys j <= i, softmax, then weighted sum of values. */
        for (int i = 0; i < CONTEXT_LENGTH; i++) {
            float q_i[HEAD_DIM];
            const float *xv = &x[i * EMBED_DIM];
            for (int d = 0; d < HEAD_DIM; d++) {
                int o = h * HEAD_DIM + d;
                q_i[d] = dot_x_int8(xv, &W_q[o * EMBED_DIM], EMBED_DIM) * s_q + b_q[o];
            }

            float A[CONTEXT_LENGTH];
            int unmasked = i + 1;
            for (int j = 0; j < unmasked; j++) {
                float acc = 0.0f;
                const float *kv = &k_h[j * HEAD_DIM];
                for (int d = 0; d < HEAD_DIM; d++) acc += q_i[d] * kv[d];
                A[j] = acc * attn_scale;
            }
            softmax_prefix(A, unmasked);

            /* Head output rows into scratch, at columns [h*HD, (h+1)*HD). */
            float *out_row = &scratch[i * EMBED_DIM + h * HEAD_DIM];
            for (int d = 0; d < HEAD_DIM; d++) {
                float acc = 0.0f;
                for (int j = 0; j < unmasked; j++) {
                    acc += A[j] * v_h[j * HEAD_DIM + d];
                }
                out_row[d] = acc;
            }
        }
    }

    /* --- out_proj + residual + layernorm 1, per position -------------- */
    for (int pos = 0; pos < CONTEXT_LENGTH; pos++) {
        float proj[EMBED_DIM];
        linear(&scratch[pos * EMBED_DIM], W_o, EMBED_DIM, EMBED_DIM,
               s_o, b_o, proj);
        float *xr = &x[pos * EMBED_DIM];
        for (int d = 0; d < EMBED_DIM; d++) xr[d] += proj[d];
        layernorm_row(xr, ln1_gamma, ln1_beta);
    }

    /* --- feed-forward + residual + layernorm 2, per position ---------- */
    for (int pos = 0; pos < CONTEXT_LENGTH; pos++) {
        float hidden[FF_DIM];
        float *xr = &x[pos * EMBED_DIM];
        linear(xr, W_ff1, EMBED_DIM, FF_DIM, s_ff1, b_ff1, hidden);
        for (int i = 0; i < FF_DIM; i++) {
            if (hidden[i] < 0.0f) hidden[i] = 0.0f; /* ReLU */
        }
        float out[EMBED_DIM];
        linear(hidden, W_ff2, FF_DIM, EMBED_DIM, s_ff2, b_ff2, out);
        for (int d = 0; d < EMBED_DIM; d++) xr[d] += out[d];
        layernorm_row(xr, ln2_gamma, ln2_beta);
    }
}

/* ----- public API ----------------------------------------------------- */

void model_forward(const uint8_t tokens[CONTEXT_LENGTH], float logits[VOCAB_SIZE]) {
    embed_tokens(tokens);
    run_layer(
        (const int8_t *)w_l0_q, s_l0_q, b_l0_q,
        (const int8_t *)w_l0_k, s_l0_k, b_l0_k,
        (const int8_t *)w_l0_v, s_l0_v, b_l0_v,
        (const int8_t *)w_l0_o, s_l0_o, b_l0_o,
        (const int8_t *)w_l0_ff1, s_l0_ff1, b_l0_ff1,
        (const int8_t *)w_l0_ff2, s_l0_ff2, b_l0_ff2,
        ln_l0_n1_gamma, ln_l0_n1_beta,
        ln_l0_n2_gamma, ln_l0_n2_beta);

    run_layer(
        (const int8_t *)w_l1_q, s_l1_q, b_l1_q,
        (const int8_t *)w_l1_k, s_l1_k, b_l1_k,
        (const int8_t *)w_l1_v, s_l1_v, b_l1_v,
        (const int8_t *)w_l1_o, s_l1_o, b_l1_o,
        (const int8_t *)w_l1_ff1, s_l1_ff1, b_l1_ff1,
        (const int8_t *)w_l1_ff2, s_l1_ff2, b_l1_ff2,
        ln_l1_n1_gamma, ln_l1_n1_beta,
        ln_l1_n2_gamma, ln_l1_n2_beta);

    /* Only the last position matters for next-token prediction. */
    const float *last = &x[(CONTEXT_LENGTH - 1) * EMBED_DIM];
    linear(last, (const int8_t *)w_output_head, EMBED_DIM, VOCAB_SIZE,  s_output_head, b_output_head, logits);
}

uint8_t model_argmax(const float logits[VOCAB_SIZE]) {
    int best = 0;
    float bestv = logits[0];
    for (int i = 1; i < VOCAB_SIZE; i++) {
        if (logits[i] > bestv) { bestv = logits[i]; best = i; }
    }
    return (uint8_t)best;
}
