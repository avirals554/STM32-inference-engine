# STM32-inference-engine

A character-level language model trained on a laptop, then running inference
on an STM32F411 with 512KB flash and 128KB RAM.
No frameworks on the microcontroller. No OS. No malloc. Just math.

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
| STM32F411CEU6 | ARM Cortex-M4, 100MHz, 512KB flash, 128KB RAM (USB-C) |
| Breadboard | for wiring everything together |
| LED monitor | to see the output |
| Jumper cables | the usual |

The STM32F411 is the "Black Pill" development board. I chose it because
it was cheap and it's one of the simplest chips that can still do
useful computation. That's it. No deeper reason.

---

## How It Works

The idea is character-level prediction, not word-level. Why? Because
word-level needs a vocabulary of tens of thousands of tokens and way
more memory than 128KB allows. Characters? There's only 65 unique ones
in the training data. That fits.

```
Training (laptop)                    Inference (STM32)
       |                                    |
       | 1. build character dictionary      | 1. load weights from flash
       |    A→0, B→1, C→2, ... (65 chars)   | 2. load dictionary
       | 2. tokenize text (tiny shakespeare) | 3. take input characters
       | 3. create embeddings (64-dim)       | 4. matrix multiply (int8)
       | 4. train with self-attention        | 5. predict next character
       | 5. quantize weights to int8         | 6. repeat
       | 6. export weights + dictionary      |
       v                                    v
   weights.bin + dict.bin ──────────> flash memory
```

### Why characters instead of words

A word-level model like GPT needs a vocabulary of 50,000+ tokens.
Each token needs an embedding vector. That's millions of parameters
before you even get to the attention layers.

A character-level model needs 65 tokens. The entire dictionary fits
in a few hundred bytes. The embeddings fit in kilobytes. The trade-off
is that the model has to learn spelling and word structure from scratch,
but that's the whole point — we want the smallest possible thing that
can still predict what comes next.

---

## The Model

The model is a two-layer transformer called `TinyTransformer`, each layer using
2-head self-attention. Everything is sized to be as small as possible while
still learning something real.

```
Input (batch of 64-char sequences)
    |
    v
Character Embedding ──── nn.Embedding(65, 64)   65 chars → 64-dim vectors
    +
Positional Embedding ─── nn.Embedding(64, 64)   position 0-63 → 64-dim vectors
    |
    v
LAYER 1
Self-Attention (2 heads)
    | Q = Linear(64→64) → reshape into 2 heads of 32
    | K = Linear(64→64) → reshape into 2 heads of 32
    | V = Linear(64→64) → reshape into 2 heads of 32
    | A = (Q @ Kᵀ) / √32
    | A = masked_fill(causal mask)    ← can't look at future characters
    | A = softmax(A)
    | output = A @ V → merge heads back → out_proj
    |
    v
Add + LayerNorm ──────── residual connection
    |
    v
Feed-Forward
    | Linear(64→128) → ReLU → Linear(128→64)
    |
    v
Add + LayerNorm ──────── residual connection
    |
    v
LAYER 2 ──────────────── same structure as Layer 1, with its own weights
    |
    v
Output Head ──────────── Linear(64→65)   → probability over 65 characters
```

### Why these numbers

| Parameter | Value | Why |
| --- | --- | --- |
| vocab size | 65 | unique characters found in tiny shakespeare |
| embedding dim | 64 | small enough for the STM32, big enough to encode patterns |
| context window | 64 | 64 characters of history to predict the next one |
| feed-forward hidden | 128 | 2x embedding dim, standard ratio |
| attention heads | 2 | the 64 dims split into 2 heads of 32 each |
| layers | 2 | two transformer blocks stacked — layer 2 refines what layer 1 found |
| batch size | 32 | 32 random windows per training step |

### Training

- **Optimizer**: Adam, learning rate 0.001
- **Loss**: CrossEntropyLoss
- **Steps**: 50,000
- **Data**: Tiny Shakespeare (~1MB, 1,003,854 training chars, 111,540 testing chars)

```
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

The training loop prints loss every 500 steps. After training, the model
generates 200 characters autoregressively — feeding each predicted character
back as input to predict the next one.

Weights are saved to `model_weights_1.pth`.

Loss dropped from ~1.5 to ~1.3 after adding multi-head attention and stacking
the second layer. Multi-head on its own barely moved it — the second layer is
what made the difference.

---

## Training Pipeline

**1. Character mapping**

Read the entire text file character by character. Every new character
gets an integer ID. Result: a dictionary of 65 entries.

```
' ' → 0
'!' → 1
...
'z' → 64
```

**2. Tokenization**

Replace every character in the text with its integer ID. The entire
text becomes a list of numbers.

**3. Train/test split**

90% training (1,003,854 chars), 10% testing (111,540 chars).

**4. Batching**

Random positions are picked from the training data. From each position,
a window of 64 characters is taken as input, and the same window
shifted by 1 is the target.

```
input:  [0.....63]     target: [1.....64]
input:  [1.....64]     target: [2.....65]
input:  [2.....65]     target: [3.....66]
...
```

32 of these windows make one batch. Random positions matter because
the model should learn language patterns, not memorize the order of
the text.

**5. Forward pass + self-attention**

Each batch goes through the transformer — embeddings, attention, feed-forward,
output head. The causal mask makes sure position N can only attend to positions
0 through N, never the future. This is what makes it a language model and not
just pattern matching.

**6. Generation**

Start with a blank context (64 zeros). Run it through the model, get a
probability distribution over 65 characters, sample one, append it to
the context, shift the window, repeat. 200 characters come out.

---

## Code Structure

```
STM32-inference-engine/
    train.py              ← full pipeline: tokenization, model, training, generation
    input.txt             ← training data (tiny shakespeare, first iteration)
    model_weights_1.pth   ← saved model weights after 50k steps
    stm32stuff/           ← the bare-metal C side (scaffolding for now, empty)
        main.c            ← inference engine entry point
        startup.c         ← startup code, runs before main
        linker.ld         ← linker script, memory layout for the chip
    .gitignore
    README.md
```

---

## Progress

How far along is this project?

```
Training pipeline (Python/PyTorch on laptop)
  [x] character dictionary        map every unique char to an integer (65 chars)
  [x] reverse dictionary          map integers back to characters
  [x] tokenization                convert full text to integer sequence
  [x] train/test split            90/10 split (1,003,854 / 111,540)
  [x] batching                    random 64-char windows, batch size 32
  [x] embedding layer             65 chars → 64-dimensional vectors
  [x] positional encoding         64 positions → 64-dimensional vectors
  [x] self-attention              Q/K/V with causal mask, 2 heads
  [x] second layer                a second transformer block, its own weights
  [x] feed-forward network        64→128→64 with ReLU
  [x] layer normalization         two LayerNorms with residual connections
  [x] training loop               50k steps, Adam, lr=0.001, CrossEntropyLoss
  [x] text generation             200 chars, autoregressive sampling
  [x] weight saving               model_weights_1.pth
  [ ] quantization                float32 → int8 for microcontroller
  [ ] weight export               dump weights + dict to C-compatible binary

Inference engine (C on STM32)
  [ ] weight loader               read weights from flash into SRAM
  [ ] dictionary loader           read char↔int mappings
  [ ] matrix multiply (int8)      the core operation, no floats
  [ ] softmax (fixed point)       probability distribution over chars
  [ ] attention (fixed point)     self-attention without floating point
  [ ] text generation loop        feed output back as input
  [ ] UART output                 print generated text over serial

Hardware
  [ ] flash the STM32             get code running on the chip
  [ ] serial monitor setup        see output from the chip
  [ ] memory profiling            does it actually fit in 128KB RAM?
  [ ] benchmark                   characters per second
```

---

## What I've Learned So Far

- An LLM is just a dictionary + weights + matrix multiplication — no magic
- Character-level models have 65 tokens vs 50,000+ for word-level — massive memory savings
- `torch.randint` for random batch positions prevents the model from memorizing text order
- `torch.stack` joins 1D tensors into a 2D batch tensor — needed because Python lists don't auto-stack
- 90/10 train/test split is the standard starting point
- The training data doesn't need to be huge — tiny shakespeare is ~1MB and that's enough to learn English character patterns
- The causal mask (`torch.tril`) is what makes a transformer a *language model* — without it, every position can see the future and there's nothing to predict
- Self-attention is just three matrix multiplications (Q, K, V) and a weighted sum — the "attention" name makes it sound more mysterious than it is
- `masked_fill` with `-inf` before softmax is how you zero out future positions — softmax turns `-inf` into 0 probability
- Residual connections (`x = x + output`) prevent the gradient from dying in deeper networks — the original signal always has a shortcut path
- LayerNorm stabilizes training by keeping activations from exploding or vanishing
- Multi-head attention by itself barely dropped the loss — stacking a second transformer layer is what took it from ~1.5 to ~1.3
- The generate function is embarrassingly simple — run the model, sample a character, shift the window, repeat
- `torch.save(model.state_dict())` saves just the learned weights, not the model code — you need the class definition to load them back

---------------------------------------

## Blogs

I'm documenting the full journey on  with more context, tangents, and the
reasoning behind every decision and the stuff that i have learned in the way they are the collection of all the things that i have realised in this 
project :

[What Happens When You Force 512KB to Think?](https://medium.com/@avirals554)

[Time When More Layers Meant Worse Model ... Birth Of Residual](https://dev.to/avirals554/time-when-more-layers-meant-worse-model-birth-of-residual-26f6)


---

*No abstractions. No frameworks. Just weights on a chip.*

~STS
