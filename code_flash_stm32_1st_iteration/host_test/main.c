/*
 * Host-side smoke test for the portable inference engine.
 *
 * Reads a seed string from stdin (or argv[1]), left-pads it with token 0 to
 * fill the context window, runs one forward pass, and prints the top-5
 * predicted next characters plus a short greedy continuation.
 *
 * Uses the same model.c / tokenizer.c that will link into the STM32 firmware,
 * so any bug we hit here will hit us on the chip too.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "model_config.h"
#include "tokenizer.h"

static void topk(const float *logits, int k, int *idx_out) {
    static int used[VOCAB_SIZE];
    for (int i = 0; i < VOCAB_SIZE; i++) used[i] = 0;
    for (int r = 0; r < k; r++) {
        int best = -1;
        float bestv = -1e30f;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            if (used[i]) continue;
            if (logits[i] > bestv) { bestv = logits[i]; best = i; }
        }
        idx_out[r] = best;
        if (best >= 0) used[best] = 1;
    }
}

int main(int argc, char **argv) {
    const char *seed;
    char stdin_buf[512];

    if (argc >= 2) {
        seed = argv[1];
    } else {
        if (!fgets(stdin_buf, sizeof(stdin_buf), stdin)) {
            fprintf(stderr, "no seed given\n");
            return 1;
        }
        seed = stdin_buf;
    }

    uint8_t seed_tokens[512];
    size_t nseed = tokenize(seed, seed_tokens, sizeof(seed_tokens));

    /* Left-pad with id 0 (which happens to be '\n' in the sorted vocab). */
    uint8_t ctx[CONTEXT_LENGTH];
    for (int i = 0; i < CONTEXT_LENGTH; i++) ctx[i] = 0;
    size_t take = nseed > CONTEXT_LENGTH ? CONTEXT_LENGTH : nseed;
    memcpy(&ctx[CONTEXT_LENGTH - take],
           &seed_tokens[nseed - take],
           take);

    float logits[VOCAB_SIZE];
    model_forward(ctx, logits);

    int top[5];
    topk(logits, 5, top);
    printf("seed tokens: %zu (used last %zu)\n", nseed, take);
    printf("top-5 next-char predictions:\n");
    for (int r = 0; r < 5; r++) {
        char buf[8];
        detokenize_one((uint8_t)top[r], buf, sizeof(buf));
        printf("  #%d id=%d logit=%.3f char=", r + 1, top[r], logits[top[r]]);
        for (const char *p = buf; *p; p++) {
            if (*p >= 32 && (unsigned char)*p < 127) putchar(*p);
            else printf("\\x%02x", (unsigned char)*p);
        }
        putchar('\n');
    }

    /* Greedy 80-char continuation. */
    printf("\ngreedy continuation:\n");
    fwrite(seed, 1, strlen(seed), stdout);
    if (strlen(seed) > 0 && seed[strlen(seed) - 1] != '\n') printf(" | ");
    for (int step = 0; step < 80; step++) {
        model_forward(ctx, logits);
        uint8_t next = model_argmax(logits);
        char buf[8];
        size_t nb = detokenize_one(next, buf, sizeof(buf));
        fwrite(buf, 1, nb, stdout);
        /* Slide the window. */
        memmove(ctx, ctx + 1, CONTEXT_LENGTH - 1);
        ctx[CONTEXT_LENGTH - 1] = next;
    }
    putchar('\n');

    return 0;
}
