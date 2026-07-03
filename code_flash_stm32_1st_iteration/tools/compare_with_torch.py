"""
Sanity check: run the fp32 PyTorch model on the same seed as host_test, print
the top-5 next-char predictions, and eyeball whether they match the quantized
C engine. Any large divergence would flag a bug in the C forward pass or in
the quantization scheme.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "model_weights_2.pth"
CORPUS_PATH = REPO_ROOT / "TinyStories-valid.txt"

CONTEXT_LENGTH = 64
EMBED_DIM = 128
NUM_HEADS = 2
HEAD_DIM = EMBED_DIM // NUM_HEADS
FF_DIM = EMBED_DIM * 2


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_embedding = nn.Embedding(CONTEXT_LENGTH, EMBED_DIM)
        self.mask = torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH))
        self.query = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.key = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.value = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.out_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.ff1 = nn.Linear(EMBED_DIM, FF_DIM)
        self.ff2 = nn.Linear(FF_DIM, EMBED_DIM)
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.query2 = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.key2 = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.value2 = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.out_proj2 = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.ff1_2 = nn.Linear(EMBED_DIM, FF_DIM)
        self.ff2_2 = nn.Linear(FF_DIM, EMBED_DIM)
        self.norm3 = nn.LayerNorm(EMBED_DIM)
        self.norm4 = nn.LayerNorm(EMBED_DIM)
        self.output_head = nn.Linear(EMBED_DIM, vocab_size)

    def attention(self, x, q, k, v, out_proj):
        B = x.shape[0]
        Q = q(x).view(B, CONTEXT_LENGTH, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        K = k(x).view(B, CONTEXT_LENGTH, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        V = v(x).view(B, CONTEXT_LENGTH, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        A = (Q @ K.transpose(-2, -1)) / HEAD_DIM**0.5
        A = A.masked_fill(self.mask == 0, float("-inf"))
        A = A.softmax(dim=-1)
        out = A @ V
        out = out.transpose(1, 2).contiguous().view(B, CONTEXT_LENGTH, EMBED_DIM)
        return out_proj(out)

    def forward(self, x):
        x = self.char_embedding(x) + self.pos_embedding(torch.arange(CONTEXT_LENGTH))
        x = self.norm1(x + self.attention(x, self.query, self.key, self.value, self.out_proj))
        x = self.norm2(x + self.ff2(torch.relu(self.ff1(x))))
        x = self.norm3(x + self.attention(x, self.query2, self.key2, self.value2, self.out_proj2))
        x = self.norm4(x + self.ff2_2(torch.relu(self.ff1_2(x))))
        return self.output_head(x)


def main():
    seed = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time, there was a"

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        chars = sorted(list(set(f.read())))
    fwd = {c: i for i, c in enumerate(chars)}
    rev = {i: c for i, c in enumerate(chars)}

    model = TinyTransformer(len(chars))
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    ids = [fwd[c] for c in seed if c in fwd]
    ctx = [0] * (CONTEXT_LENGTH - len(ids)) + ids[-CONTEXT_LENGTH:]
    ctx_t = torch.tensor(ctx, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(ctx_t)[0, -1]

    topv, topi = logits.topk(5)
    print(f"fp32 top-5 for seed {seed!r}:")
    for v, i in zip(topv, topi):
        ch = rev[i.item()]
        show = repr(ch) if not ch.isprintable() else ch
        print(f"  id={i.item()} logit={v.item():.3f} char={show}")


if __name__ == "__main__":
    main()
