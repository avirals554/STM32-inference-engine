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
more memory than 128KB allows. Characters? There's only about 80 of
them. That fits.

```
Training (laptop)                    Inference (STM32)
       |                                    |
       | 1. build character dictionary      | 1. load weights from flash
       |    A→0, B→1, C→2, ... (~80 chars)  | 2. load dictionary
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

A character-level model needs ~80 tokens. The entire dictionary fits
in a few hundred bytes. The embeddings fit in kilobytes. The trade-off
is that the model has to learn spelling and word structure from scratch,
but that's the whole point — we want the smallest possible thing that
can still predict what comes next.

---

## Training

Training data: [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
(~1MB of Shakespeare's plays as raw text)

```
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

The training pipeline so far:

**1. Character mapping**

Read the entire text file character by character. Every new character
gets an integer ID. Result: a dictionary of ~80 entries.

```
A → 0
B → 1
C → 2
...
```

**2. Tokenization**

Replace every character in the text with its integer ID. The entire
text becomes a list of numbers.

**3. Train/test split**

90% training, 10% testing. Standard split.

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

---

## Code Structure

```
STM32-inference-engine/
    train.py       ← training pipeline (tokenization, batching, embeddings)
    input.txt      ← training data (tiny shakespeare)---> this is just for the first iteration that is done right now we will hopefully change it 
    README.md
```

---

## Progress

How far along is this project?

```
Training pipeline (Python/PyTorch on laptop)
  [x] character dictionary        map every unique char to an integer
  [x] reverse dictionary          map integers back to characters
  [x] tokenization                convert full text to integer sequence
  [x] train/test split            90/10 split
  [x] batching                    random 64-char windows, batch size 32
  [ ] embedding layer             64-dimensional vectors per character
  [ ] positional encoding         so the model knows character order
  [ ] self-attention              the actual "learning" mechanism
  [ ] training loop               forward pass, loss, backprop
  [ ] quantization                float32 → int8 for microcontroller
  [ ] weight export               dump weights + dict to binary files

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
- Character-level models have ~80 tokens vs 50,000+ for word-level — massive memory savings
- `torch.randint` for random batch positions prevents the model from memorizing text order
- `torch.stack` joins 1D tensors into a 2D batch tensor — needed because Python lists don't auto-stack
- 90/10 train/test split is the standard starting point
- The training data doesn't need to be huge — tiny shakespeare is ~1MB and that's enough to learn English character patterns

---

## Blog

I'm documenting the full journey on Medium with more context, tangents, and the
reasoning behind every decision:

[What Happens When You Force 512KB to Think?](https://medium.com/@avirals554)

---

*No abstractions. No frameworks. Just weights on a chip.*

~STS
