/*
 * Application entry point for the STM32F411 next-word prediction demo.
 *
 * REPL loop:
 *   1. Print a prompt on USART1.
 *   2. Read a line of seed text (ending in CR or LF).
 *   3. Tokenize + left-pad to CONTEXT_LENGTH.
 *   4. Run model_forward, greedy-sample the next character.
 *   5. Slide the window and print another N characters.
 *
 * Everything is polling / straight-line -- no RTOS, no interrupts.
 */

#include <stdint.h>
#include <string.h>

#include "model.h"
#include "model_config.h"
#include "oled.h"
#include "tokenizer.h"
#include "uart.h"

#define GEN_CHARS       120     /* how many chars to generate per prompt */
#define SEED_BUF_LEN    256

static uint8_t ctx[CONTEXT_LENGTH];
static float   logits[VOCAB_SIZE];
static char    seed[SEED_BUF_LEN];

static void read_line(char *dst, size_t cap) {
    size_t n = 0;
    for (;;) {
        uint8_t c = uart_getc();
        if (c == '\r' || c == '\n') {
            uart_puts("\r\n");
            break;
        }
        if (c == 0x7F || c == 0x08) {   /* backspace / DEL */
            if (n > 0) { n--; uart_puts("\b \b"); }
            continue;
        }
        if (n + 1 < cap && c >= 0x20) {
            dst[n++] = (char)c;
            uart_putc((char)c);          /* echo */
        }
    }
    dst[n] = '\0';
}

static void load_context(const char *text) {
    uint8_t tokens[SEED_BUF_LEN];
    size_t nseed = tokenize(text, tokens, sizeof(tokens));

    for (int i = 0; i < CONTEXT_LENGTH; i++) ctx[i] = 0;
    size_t take = nseed > CONTEXT_LENGTH ? CONTEXT_LENGTH : nseed;
    memcpy(&ctx[CONTEXT_LENGTH - take],
           &tokens[nseed - take],
           take);
}

/* Baked-in seed that runs automatically once at boot so you can verify
 * the OLED + inference pipeline without any UART adapter. */
static const char *BOOT_SEED = "hello my name is ";

static void run_generation(const char *seed_str) {
    load_context(seed_str);

    uart_puts(seed_str);
    uart_puts(" | ");

    /* Screen: fresh page with the seed, then live-append predictions. */
    oled_clear();
    oled_puts(seed_str);
    oled_puts("|");

    for (int step = 0; step < GEN_CHARS; step++) {
        model_forward(ctx, logits);
        uint8_t next = model_argmax(logits);

        char out[8];
        size_t n = detokenize_one(next, out, sizeof(out));
        for (size_t i = 0; i < n; i++) {
            if (out[i] == '\n') uart_putc('\r');
            uart_putc(out[i]);
            oled_putc(out[i]);
        }

        memmove(ctx, ctx + 1, CONTEXT_LENGTH - 1);
        ctx[CONTEXT_LENGTH - 1] = next;
    }
}

int main(void) {
    uart_init();
    oled_init();

    uart_puts("\r\n");
    uart_puts("TinyTransformer inference engine (2-layer, int8) on STM32F411\r\n");
    uart_puts("Vocab=98 chars, context=64. Type a seed and press Enter.\r\n");

    /* Auto-run: show something on the OLED without needing an adapter. */
    run_generation(BOOT_SEED);

    for (;;) {
        uart_puts("\r\n>>> ");
        read_line(seed, sizeof(seed));
        if (seed[0] == '\0') continue;
        run_generation(seed);
    }
}
