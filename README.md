# How to run after you git pull from this code -
st-flash --reset write inference.bin 0x08000000

we are essentially flashing the .bin file at the address 0x0800000 and i have also speicified the adresses of all the data that are used in this code in the code flash stm 32 folder if you want to change the architecture and use another memory location you can do it too .

# STM32-inference-engine

A character-level language model trained on a laptop, then running inference
on an STM32F411 with 512KB flash and 128KB RAM.
No frameworks on the microcontroller. No OS. No malloc. Just math.

**It works.** The int8 engine on the chip matches the fp32 PyTorch model's
top-5 predictions exactly, and the board generates coherent English on a
little OLED. Proof is in the [Validation](#validation) section.

### Demo

[![Watch the STM32 running inference](https://img.youtube.com/vi/2sFopyFJplE/maxresdefault.jpg)](https://youtu.be/2sFopyFJplE)

*The Black Pill generating text character by character on the OLED — click to watch.*

---

## What This Is

An LLM is just weights and a dictionary. That's it. The "intelligence" is
matrix multiplication applied to numbers that came out of training.

so the question is  if the math is that simple, can we
run it on the cheapest, smallest hardware we can find? i dont mean that every
microcontroller that we find in fridges needs AI, but because *can* it run inference as an experiment ?, is
the underlying math really is as simple as we claim ?

The chip does inference only. Training happens on my Mac. The inference
engine on the chip is written from scratch in C — no libraries, no
floating point luxury, no abstractions.

---

## Hardware

| Part | What It Is |
| --- | --- |
| STM32F411CEU6 | ARM Cortex-M4, 100MHz, 512KB flash, 128KB RAM (USB-C) — the "Black Pill" |
| SSD1306 OLED | tiny I2C display, shows the generated text |
| USB-serial adapter | for typing the seed prompt over UART |
| Breadboard + jumper cables | the usual |

Wiring: OLED on `SCL→PB6`, `SDA→PB7`, `VCC→3V3`, `GND→GND`.
Serial on `PA9 (TX)→adapter RX`, `PA10 (RX)→adapter TX`, `GND→GND`, at `115200 8N1`.

The STM32F411 was chosen because it was cheap and it's one of the simplest
chips that can still do useful computation. That's it. No deeper reason.

---

## How It Works

Character-level prediction, not word-level. Word-level needs a vocabulary of
tens of thousands of tokens and way more memory than 128KB allows. Characters?
There are only 98 unique ones in the training data. That fits.

```
Training (laptop)                    Inference (STM32)
       |                                    |
       | 1. build character dictionary      | 1. weights already in flash (int8)
       |    (98 chars from TinyStories)     | 2. read seed prompt over UART
       | 2. tokenize text                    | 3. run 2-layer transformer (float32)
       | 3. train 2-layer transformer        | 4. argmax → next character
       | 4. quantize weights to int8         | 5. print to OLED, slide window
       | 5. export weights + tokenizer → .h  | 6. repeat
       v                                    v
   generated C headers ──────────────> compiled into firmware → flash
```

### Why characters instead of words

A word-level model like GPT needs a vocabulary of 50,000+ tokens. Each token
needs an embedding vector — millions of parameters before you even reach the
attention layers.

A character-level model needs 98 tokens. The entire dictionary fits in a few
hundred bytes. The trade-off is that the model learns spelling and word
structure from scratch, but that's the whole point — the smallest possible
thing that can still predict what comes next.

### Why TinyStories instead of Shakespeare

The first iteration trained on Tiny Shakespeare and the output was rough —
Shakespeare's English is complex and archaic, too much for a model this small
to imitate well. Switching to
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — simple
English written for small kids — the model started producing actual grammatical
sentences. The dataset does a lot of the heavy lifting: simpler language means
a tiny model can still sound coherent.

---

## The Model

A two-layer transformer called `TinyTransformer`, each layer using 2-head
self-attention. Sized to be as small as possible while still learning something
real.

```
Input (batch of 64-char sequences)
    |
    v
Character Embedding ──── nn.Embedding(98, 128)   98 chars → 128-dim vectors
    +
Positional Embedding ─── nn.Embedding(64, 128)   position 0-63 → 128-dim vectors
    |
    v
LAYER 1
Self-Attention (2 heads)
    | Q = Linear(128→128) → reshape into 2 heads of 64
    | K = Linear(128→128) → reshape into 2 heads of 64
    | V = Linear(128→128) → reshape into 2 heads of 64
    | A = (Q @ Kᵀ) / √64
    | A = masked_fill(causal mask)    ← can't look at future characters
    | A = softmax(A)
    | output = A @ V → merge heads back → out_proj
    |
    v
Add + LayerNorm ──────── residual connection
    |
    v
Feed-Forward
    | Linear(128→256) → ReLU → Linear(256→128)
    |
    v
Add + LayerNorm ──────── residual connection
    |
    v
LAYER 2 ──────────────── same structure as Layer 1, with its own weights
    |
    v
Output Head ──────────── Linear(128→98)   → probability over 98 characters
```

### Why these numbers

| Parameter | Value | Why |
| --- | --- | --- |
| vocab size | 98 | unique characters found in TinyStories |
| embedding dim | 128 | doubled from iteration 1's 64 for higher resolution |
| context window | 64 | 64 characters of history to predict the next one |
| feed-forward hidden | 256 | 2x embedding dim, standard ratio |
| attention heads | 2 | the 128 dims split into 2 heads of 64 each |
| layers | 2 | two transformer blocks stacked — layer 2 refines what layer 1 found |
| batch size | 32 | 32 random windows per training step |

### Training

- **Optimizer**: Adam, learning rate 0.001
- **Loss**: CrossEntropyLoss
- **Steps**: 50,000
- **Data**: TinyStories (`TinyStories-valid.txt`) — ~17.5M training chars (90%), ~1.9M testing chars (10%)

After training, the weights are saved to `model_weights_2.pth`, then quantized
to int8 and exported as C headers.

---

## Training Pipeline

**1. Character mapping** — read the text character by character, every unique
character gets an integer ID. 98 entries.

**2. Tokenization** — replace every character with its integer ID. The whole
text becomes a list of numbers.

**3. Train/test split** — 90% training, 10% testing.

**4. Batching** — random positions in the training data, each giving a 64-char
input window and the same window shifted by 1 as the target. 32 windows per
batch. Random positions matter so the model learns language patterns, not the
order of the text.

```
input:  [0.....63]     target: [1.....64]
input:  [1.....64]     target: [2.....65]
...
```

**5. Forward pass** — each batch goes through both transformer layers. The
causal mask ensures position N only attends to positions 0 through N. This is
what makes it a language model.

**6. Generation** — start from a context, run the model, sample the next
character, slide the window, repeat.

> Note: the Python training script samples with `torch.multinomial`
> (probabilistic). The on-chip C engine uses greedy `argmax` — deterministic,
> which is what makes it directly comparable to the reference (below).

---

## Quantization

The trained weights are float32. Flash is precious, so `tools/quantize.py`
compresses them to int8 with a **per-tensor symmetric** scheme:

```
scale   = max(|W|) / 127
W_int8  = round(W / scale)   clipped to [-127, 127]
W_float = W_int8 * scale     ← reconstructed at inference time
```

Weights and embedding tables become int8. Biases and LayerNorm gamma/beta stay
float32 — they're small, they accumulate the int8×float products, and keeping
them precise avoids error blow-up for basically no flash cost.

The quantizer emits three headers into `firmware/generated/`:
`model_config.h`, `model_weights.h`, `tokenizer_data.h`.

---

## The Inference Engine (C on STM32)

The engine (`firmware/src/model.c`) is pure, portable C — no STM32 or HAL
dependencies. The *same* object file links into both the on-chip firmware and a
host-side test harness, so any bug reproduces on the laptop instead of forcing
you to debug blind on the chip.

Only the stored weights are int8; all activation math runs in float32 on the
Cortex-M4's hardware FPU. Quantization here is about shrinking flash, not
avoiding floating point.

### Memory budget

The forward pass runs on four static buffers in `.bss`:

```
x       : 32 KB   hidden state, persists across layers   (64 × 128 floats)
scratch : 32 KB   attention concat / feed-forward scratch (64 × 128 floats)
k_h     : 16 KB   per-head keys, one head at a time       (64 × 64 floats)
v_h     : 16 KB   per-head values, one head at a time     (64 × 64 floats)
---------------
total   : 96 KB   ← fits in the STM32F411's 128 KB SRAM
```

### Footprint (from `arm-none-eabi-size`)

| Section | Size | Chip capacity |
| --- | --- | --- |
| flash | 306 KB | 512 KB |
| SRAM | 100 KB | 128 KB |

Comfortable margins on both. No heroics required.

### On the chip

`main.c` runs a small UART REPL: read a seed line at 115200 baud, pad it to 64
tokens, then loop — forward pass → argmax → print the character to the OLED →
slide the window. Reset the board and it prints:

```
TinyTransformer inference engine (2-layer, int8) on STM32F411
Vocab=98 chars, context=64. Type a seed and press Enter.

>>>
```

---

## Validation

The question that matters: does the int8 C engine actually match the fp32
PyTorch model, or does it just produce plausible-looking gibberish?

`host_test/` compiles the same `model.c` on the laptop, and
`tools/compare_with_torch.py` runs the fp32 PyTorch reference on the same seed.
On the seed `"Once upon a time, there was a"`:

- The **top-5 next-character predictions match exactly**, with logit
  differences around **0.1** — quantization loss is well inside the model's own
  noise floor.
- Greedy continuation:
  `Once upon a time, there was a little girl named Lily who loved to play ...`

Coherent English out of a 306KB flash image. Not luck — verified.

> Caveat worth keeping honest: this is a single-seed, top-5 check. It's
> convincing, but a fully bulletproof test would run many seeds and measure
> logit error across the whole vocabulary, not just the top-5 on one prompt.

---

## Code Structure

```
STM32-inference-engine/
    train_iteration1.py      ← first model (Tiny Shakespeare, 64-dim)
    train_iteration2.py      ← current model (TinyStories, 128-dim, vocab 98)
    weights.py               ← weight inspection / helpers
    test_data.py             ← data checks
    model_weights_1.pth      ← iteration 1 weights
    model_weights_2.pth      ← iteration 2 weights (the one that gets flashed)
    tools.md                 ← notes
    input.txt                ← Tiny Shakespeare (iteration 1 data)

    code_flash_stm32_1st_iteration/
        README.md            ← detailed build + wiring guide
        tools/
            quantize.py          fp32 .pth → int8 C headers
            compare_with_torch.py fp32 vs int8 sanity check
        firmware/
            inc/                 model.h, tokenizer.h, uart.h, oled.h, i2c.h
            src/                 model.c, tokenizer.c, uart.c, oled.c,
                                 i2c.c, main.c, startup.c
            generated/           model_config.h, model_weights.h,
                                 tokenizer_data.h  (from quantize.py)
            stm32f411.ld         linker script
            Makefile
        host_test/               compiles model.c on the dev host
            main.c
            Makefile
```

`model.c` and `tokenizer.c` are pure C and link into both the firmware and the
host test — so bugs are reproducible on the laptop.

---

## Building & Running

```
# 1. regenerate weight headers (only after retraining)
cd code_flash_stm32_1st_iteration
python3 tools/quantize.py

# 2. host smoke test (runs the real engine on your laptop)
cd host_test && make
./host_test "Once upon a time, there was a"

# 3. flash the chip (needs arm-none-eabi-gcc + st-flash)
cd ../firmware && make        # → inference.elf, inference.bin, prints footprint
make flash                    # writes to 0x08000000 via STLink
```

Full wiring and build details are in
[`code_flash_stm32_1st_iteration/README.md`](code_flash_stm32_1st_iteration/README.md).

---

## Progress

```
Training pipeline (Python/PyTorch on laptop)
  [x] character dictionary        map every unique char to an integer (98 chars)
  [x] tokenization                convert full text to integer sequence
  [x] train/test split            90/10 split (~17.5M / ~1.9M chars)
  [x] batching                    random 64-char windows, batch size 32
  [x] embedding + positional       98 chars, 64 positions → 128-dim vectors
  [x] self-attention              Q/K/V with causal mask, 2 heads
  [x] second layer                a second transformer block, its own weights
  [x] feed-forward + layernorm    128→256→128 with ReLU, residual + norm
  [x] training loop               50k steps, Adam, lr=0.001, CrossEntropyLoss
  [x] TinyStories switch          simpler English → coherent output
  [x] text generation             autoregressive sampling

Quantization + export (tools/)
  [x] int8 quantization           per-tensor symmetric, scale = max|W|/127
  [x] weight export               weights + tokenizer → C headers
  [x] torch comparison            fp32 vs int8, top-5 match, logit Δ ~0.1

Inference engine (C on STM32)
  [x] portable model.c            links into both firmware and host test
  [x] matrix multiply             int8 weights × float32 activations
  [x] softmax                     numerically stable, prefix-masked
  [x] attention                   per-head K/V precompute, causal mask
  [x] text generation loop        forward → argmax → slide window
  [x] tokenizer                   UTF-8 aware (14 non-ASCII chars)
  [x] UART REPL                   type a seed at 115200 baud
  [x] OLED output                 SSD1306 over I2C

Hardware
  [x] flash the STM32             306KB of 512KB flash used
  [x] serial + OLED setup         working REPL + display
  [x] memory profiling            100KB of 128KB SRAM — it fits
  [ ] benchmark                   characters per second on-chip
```

---

## What I've Learned So Far

- An LLM is just a dictionary + weights + matrix multiplication — no magic
- Character-level models have ~98 tokens vs 50,000+ for word-level — massive memory savings
- The causal mask (`torch.tril`) is what makes a transformer a *language model* — without it, every position can see the future and there's nothing to predict
- Self-attention is just three matrix multiplications (Q, K, V) and a weighted sum — the name makes it sound more mysterious than it is
- Residual connections (`x = x + output`) give the original signal a shortcut path so the gradient doesn't die in deeper networks
- Multi-head attention by itself barely dropped the loss — stacking a second transformer layer is what took it from ~1.5 to ~1.3
- The dataset matters more than I expected — swapping Shakespeare for TinyStories (simpler English) is what made the output actually readable
- int8 weight-only quantization (per-tensor, `scale = max|W|/127`) loses almost nothing — the top-5 predictions match fp32 *exactly*, logit deltas ~0.1
- Keeping biases and LayerNorm params in float32 while quantizing the weights avoids error blow-up for basically free — they're tiny
- Only the stored weights are int8 — activations stay float32 on the Cortex-M4 FPU. Quantization shrinks flash, it doesn't avoid float math
- Linking the same `model.c` into both the host test and the firmware means any bug reproduces on the laptop — you never debug blind on the chip
- The whole thing is 306KB flash and 100KB SRAM — it genuinely fits with room to spare

---

## Blogs

I'm documenting the full journey on  with more context, tangents, and the
reasoning behind every decision and the stuff that i have learned in the way they are the collection of all the things that i have realised in this 
project :

[What Happens When You Force 512KB to Think?](https://medium.com/@avirals554)

[Time When More Layers Meant Worse Model ... Birth Of Residual](https://dev.to/avirals554/time-when-more-layers-meant-worse-model-birth-of-residual-26f6)


---

*No abstractions. No frameworks. Just weights on a chip.*

~STS
