# TinyTransformer Inference on STM32F411CEU6 (WeAct Black Pill)

Runs the 2-layer, 128-embed, 2-head character transformer trained in
`../train_iteration2.py` directly on a Cortex-M4 with 512 KB flash and
128 KB SRAM. Weights are int8-quantized (weight-only, per-tensor); all
activation math stays in float32 on the hardware FPU.

## Footprint

| Section | Size    | Chip capacity |
|---------|---------|---------------|
| flash   | 306 KB  | 512 KB        |
| SRAM    | 100 KB  | 128 KB        |

The extra ~1 KB SRAM over the UART-only version is the SSD1306 framebuffer.

Breakdown (from `arm-none-eabi-size`):

```
text = 311,536   (weights + code, .rodata + .text)
data =      80   (initialized RAM globals)
bss  = 101,064   (activation buffers + tokenizer state)
```

The 96 KB of activation buffers are exactly what the arithmetic requires:
`x[64][128] + scratch[64][128] + k_h[64][64] + v_h[64][64]` = 32 + 32 + 16 + 16.

## Layout

```
claude-1st-iteration/
├── model_weights_2.pth              (copied from the repo root)
├── tools/
│   ├── quantize.py                  fp32 .pth  ->  int8 C headers
│   └── compare_with_torch.py        fp32 vs int8 sanity check
├── firmware/
│   ├── inc/                         model.h, tokenizer.h, uart.h
│   ├── src/                         model.c, tokenizer.c, uart.c,
│   │                                main.c, startup.c
│   ├── generated/                   model_config.h, model_weights.h,
│   │                                tokenizer_data.h  (from quantize.py)
│   ├── stm32f411.ld                 linker script
│   └── Makefile
└── host_test/                       compiles model.c on the dev host
    ├── main.c
    └── Makefile
```

`firmware/src/model.c` and `firmware/src/tokenizer.c` are pure C and
have no STM32 dependencies -- they link into both the on-chip firmware
and the host-side smoke test, so any bug reproduces on the dev machine.

## Building

### 1. Regenerate weight headers (only after retraining)

```
cd claude-1st-iteration
python3 tools/quantize.py
```

Produces `firmware/generated/{model_config,model_weights,tokenizer_data}.h`.

### 2. Host smoke test

```
cd host_test
make
./host_test "Once upon a time, there was a"
```

Expected: top-5 next chars roughly matching the fp32 PyTorch reference and
a coherent short continuation. Sample run of the current weights:

```
top-5 next-char predictions:
  #1 id=1  logit=10.132 char= (space)
  #2 id=70 logit=6.027  char=n
  #3 id=68 logit=3.971  char=l
  #4 id=76 logit=3.421  char=t
  #5 id=75 logit=3.029  char=s

greedy continuation:
Once upon a time, there was a |  little girl named Lily who loved to play ...
```

Compared against `tools/compare_with_torch.py` on the same seed, the top-5
ids match exactly with logit deltas around 0.1 -- quantization loss is
well inside the noise floor of the trained model.

### 3. STM32 firmware

Requirements: `arm-none-eabi-gcc` (Arm GNU Toolchain 15.x tested) and
`st-flash` (from stlink-tools) for flashing.

```
cd firmware
make               # -> inference.elf, inference.bin, prints footprint
make flash         # writes inference.bin to 0x08000000 via STLink
```

## Running on hardware

1. Flash the board (STLink SWD connector on the WeAct Black Pill).
2. Connect a USB-serial adapter (for typing the seed):
   - PA9  (chip TX) -> adapter RX
   - PA10 (chip RX) -> adapter TX
   - GND -> GND
3. Connect the SSD1306 OLED (for displaying the output):
   - VCC -> 3V3
   - GND -> GND
   - SCL -> PB6
   - SDA -> PB7
4. Open a terminal at `115200 8N1` (e.g. `screen /dev/tty.usbserial-XXXX 115200`).
5. Reset the chip. You should see:

```
TinyTransformer inference engine (2-layer, int8) on STM32F411
Vocab=98 chars, context=64. Type a seed and press Enter.

>>>
```

Type a seed (e.g. "Once upon a time"), press Enter, and the chip will
print `seed | <120-char greedy continuation>`.

## How the model maps onto the chip

- **Weights in flash.** Every weight matrix is quantized to int8 with a
  per-tensor float scale (`quantize.py`). Biases and LayerNorm gamma/beta
  remain float32 -- they're small and act as accumulators. Embedding
  tables are also int8-quantized.
- **Activations in SRAM.** The transformer forward pass runs on two
  32 KB float buffers (`x`, `scratch`) plus a 16 KB `k_h`/`v_h` pair
  used one attention head at a time. FF hidden and out_proj scratch
  are per-position local variables on the stack (~1.5 KB).
- **Attention scaling** uses `1/sqrt(HEAD_DIM)`; the causal mask is done
  by only computing/softmaxing positions `<= i`, which also saves
  roughly half the score compute at the last few positions.
- **Prediction API.** `model_forward(tokens, logits)` writes only the
  logits for the last position (all we ever sample). `model_argmax`
  picks the greedy next token.
- **UART REPL.** `main.c` reads a line from USART1 at 115200 baud, pads
  to 64 tokens, then loops: forward -> argmax -> print -> slide window.

## Retraining / vocabulary changes

`train_iteration2.py` builds the vocabulary from `TinyStories-valid.txt`,
so `quantize.py` reads the *same* file to reproduce the char-to-id map.
If you retrain on a different corpus, keep the training script and
`quantize.py` pointed at the same source and re-run both. The
`VOCAB_SIZE = 98` assertion in `quantize.py` will loudly fail if the
vocab changes size, and you can bump the constant in step.
