# references - github.com/avirals554/STM32-inference-engines
# important facts that needs to be remembered for checking the code and doing stuff -
# the size of the dictionary is - 98 --forward_map (TinyStories has way more unique chars than the shakespeare input.txt did)
# the size of the training dataset- ~17.5 million chars (90% of TinyStories-valid.txt)
# the size of the testing dataset- ~1.9 million chars (10% of TinyStories-valid.txt)
# the accuracy was not good in iteration1 because the text (shakespeare) was too complex english, so now we are using TinyStories
# which is much simpler english written for small kids -- https://huggingface.co/datasets/roneneldan/TinyStories


import torch
import torch.nn as nn

# ===== CONFIG =====
# unlike iteration1 where all dimentions were hardcoded as magic numbers (64,65,128), here i pulled them out into variables at the top
# so that if i want to change the model size i only change it in ONE place and everything else uses the variable.
# what each one does --
# context_length -- sequence length (chars per training example), NOT vocab size. how many chars the model looks at in one go
# embed_dim      -- the dimention of the embedding vector for each character. iteration1 was 64, doubling it for higher resolution
# num_heads      -- number of attention heads (same as iteration1). splits the embed_dim into this many parallel attention streams
# head_dim       -- dimention PER head. computed, not hardcoded. comes out to 64 here (128/2)
# batch_size     -- how many sequences in one training batch (same as iteration1)
# ff_dim         -- the feed-forward expansion. we enlarge to this then shrink back. 128->256->128 here
# learning_rate  -- learning rate for Adam optimizer (same as iteration1)
# training_steps -- number of training iterations (same as iteration1)
# btw the vocab size for the tiny stories data set is like 92 words for the embeddings table
#
context_length = 64  # sequence length (chars per training example), NOT vocab size
embed_dim = 128
num_heads = 2
head_dim = embed_dim // num_heads
batch_size = 32
ff_dim = embed_dim * 2
learning_rate = 0.001
training_steps = 50000

# ===== TOKENISER =====
# this is gonna take the txt file and iterate it one character at a time and then make a dictionary that we are gonna call forward_map()
# and this is gonna contain all character that are in this .txt file
# the ids that are given are just simple integers that are incremented by one for each new character
# Example-- 'A'-->0,'B'-->1,'C'-->2......
# the reverse_map just reverces the dictionary so that the keys become the value and the values become the keys --
# Example - 0-->'A',1-->'B',2-->'C'.......
# i shortened iteration1's mapping/reverse_mapping/tokenise_local into a few lines using set() and dict comprehensions but its the SAME idea
# same idea as iteration1: build forward_map char->id and reverse_map id->char
with open("TinyStories-valid.txt", "r", encoding="utf-8") as f:
    text = f.read()

# set() gives unique chars, sorted() makes the ids deterministic across runs
chars = sorted(list(set(text)))
# number of unique characters -- NOT hardcoded to 65 like iteration1, computed from the file (=98 here)
vocab_size = len(chars)


# char -> id
forward_map = {c: i for i, c in enumerate(chars)}
# id -> char (used in generate() to turn predictions back into text)
reverse_map = {i: c for i, c in enumerate(chars)}
print(f"vocab_size: {vocab_size}")

# the entire file converted to a list of ids
tokenised_text = [forward_map[c] for c in text]
# after the data is ready we are gonna use the torch library finally
# this step is important cause now we can actually do stuff with the data , in python to perform any action we must iterate the entire list and then do stuff with it but pytorch is
# is just annoyingly useful in this case .
data = torch.tensor(tokenised_text)
split_point = int(0.9 * len(data))
training_data = data[:split_point]
testing_data = data[split_point:]


# ===== MODEL =====
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # setting the constructor for the initial values that we are every gonna need for the training of the data
        # lookup table -- one embed_dim vector per character (98 x 128)
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        # lookup table -- one embed_dim vector per position (64 x 128)
        self.pos_embedding = nn.Embedding(context_length, embed_dim)
        # lower-triangular causal mask, prevents looking at future chars
        self.mask = torch.tril(torch.ones(context_length, context_length))
        # layer 1 -- attention block + feed-forward block
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        # mixes the 2 heads info together after attention
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # these are for changing the dimentions we are doing this to enlarge the matrix as to make it of higher resolution so as to make the
        # data and weights more refined
        self.ff1 = nn.Linear(embed_dim, ff_dim)
        # this is to join them back again
        self.ff2 = nn.Linear(ff_dim, embed_dim)
        # normalization after attention -- improves accuracy a lot
        self.norm1 = nn.LayerNorm(embed_dim)
        # normalization after feed-forward
        self.norm2 = nn.LayerNorm(embed_dim)
        # this here i am adding the variables for the 2nd attention block for lower loss function ...
        # SAME shapes as layer 1 but separate weights -- layer 2 has its OWN parameters
        self.query2 = nn.Linear(embed_dim, embed_dim)
        self.key2 = nn.Linear(embed_dim, embed_dim)
        self.value2 = nn.Linear(embed_dim, embed_dim)
        self.out_proj2 = nn.Linear(embed_dim, embed_dim)
        self.ff1_2 = nn.Linear(embed_dim, ff_dim)
        self.ff2_2 = nn.Linear(ff_dim, embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        # output_head only runs ONCE at the very end, converts embed_dim -> vocab_size (one score per character)
        self.output_head = nn.Linear(embed_dim, vocab_size)

    # in iteration1 i wrote the entire attention block out TWICE (once for layer 1, once for layer 2). it was the same code copy-pasted.
    # here i pulled it into a helper function so both layers call it with their own q,k,v,out_proj weights.
    # this is the start of the attention stuff i am writting this as a way to seperate the code in section inside a functions
    def attention(self, x, q, k, v, out_proj):
        # getting the dimentions of the 2d array (B = batch size)
        B = x.shape[0]
        Q = q(x).view(B, context_length, num_heads, head_dim).transpose(1, 2)
        K = k(x).view(B, context_length, num_heads, head_dim).transpose(1, 2)
        V = v(x).view(B, context_length, num_heads, head_dim).transpose(1, 2)
        A = (Q @ K.transpose(-2, -1)) / head_dim**0.5
        A = A.masked_fill(self.mask == 0, float("-inf"))
        # the -1 is to apply softmax across each row
        At = A.softmax(dim=-1)
        out = At @ V
        # joining the 2 heads back together into one embed_dim wide vector
        out = out.transpose(1, 2).contiguous().view(B, context_length, embed_dim)
        # mixes the 2 heads info together
        return out_proj(out)

    def forward(self, x):
        # feed forward function
        # add the char embedding and the position embedding together -- the model needs to know WHAT the char is AND WHERE it is
        x = self.char_embedding(x) + self.pos_embedding(torch.arange(context_length))
        # ===== LAYER 1 =====
        # attention -> residual add -> normalize
        # added the normalization + residual to improve accuracy. loss dropped a lot after this
        x = self.norm1(
            x + self.attention(x, self.query, self.key, self.value, self.out_proj)
        )
        # feed forward -- enlarge to ff_dim for higher resolution / refined data, then join back to embed_dim, then residual + norm
        x = self.norm2(x + self.ff2(torch.relu(self.ff1(x))))
        # ===== LAYER 2 =====
        # this is the 2nd attention block, it works on what layer 1 already figured out
        # the data (context matrix) flows from layer 1 into here, layer 2 has its OWN weights
        x = self.norm3(
            x + self.attention(x, self.query2, self.key2, self.value2, self.out_proj2)
        )
        x = self.norm4(x + self.ff2_2(torch.relu(self.ff1_2(x))))
        # ===== OUTPUT =====
        # output_head only runs ONCE at the very end, converts embed_dim dims -> vocab_size (one score per character)
        return self.output_head(x)


def get_batch():
    # so this is function is simple we are gonna take the training data list and then we are gonna make "batches" of it the idea is that that
    # batches would look something like this --
    # [0.....64] : [1.....65] , [1.....65]:[2......66],[2.......66]:[3.......67]..... and so on .
    # so this was interesting right cause in this we are choosing random numbers or indices from the list and then we are gonna use that positions
    # and have a data set of context_length FROM that index and then make the collection of those , and random thing is important so that the llm doesnt just
    # remembers that patter of the txt itself but actually learn the meaning and how to form sentences .
    positions = torch.randint(len(training_data) - context_length, (batch_size,))
    input_vector = [training_data[i : i + context_length] for i in positions]
    value_vector = [training_data[i + 1 : i + context_length + 1] for i in positions]
    # the reason that we are adding stack method here is cause we made the arrays of stuff in the previous step and then we want to join those arrays as well
    # this got a bit confusing but essentially what is haapening is this --
    # we dont declare if the vector is 2d or 3d hence i am guessing it can become 2d its not like vector<vector<int>> so that would mean that
    # you need to joing the 1D vectors to make a 2D vector and in order to do that we use .stack() method
    return torch.stack(input_vector), torch.stack(value_vector)


# ===== TRAINING =====
# this is the part for the declaration of the model or the tiny transformer
model = TinyTransformer()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for step in range(training_steps):
    # calling the function that we made for the getting the set of the inputs and the outputs
    inputs, targets = get_batch()
    predictions = model(inputs).view(-1, vocab_size)
    targets = targets.view(-1)
    loss = loss_fn(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"step {step}, loss: {loss.item():.4f}")


# ===== GENERATION =====
# ok lets take a look at this function and see what it does alright ?
def generate(model, max_chars=200):
    # so the torch.zeros create a tensor that is filled with the scalar 0
    # in this there would be the integer of 0 in each of the cloumn and rows of the tensor
    # (note -- 0 happens to map to whichever char is alphabetically first in our sorted chars list, usually '\n' or ' ')
    context = torch.zeros(1, context_length, dtype=torch.long)
    result = []
    for _ in range(max_chars):
        output = model(context)
        # if the weights are some random data that we change to move closer to the truth then
        # why did we initialize it to 0?
        # take the LAST position's logits, softmax to get probabilities over all vocab
        probs = torch.softmax(output[0, -1], dim=0)
        # sample one char from the probability distribution (not argmax -- adds randomness)
        next_char = torch.multinomial(probs, 1).item()
        result.append(reverse_map[next_char])
        # slide the context window -- drop the oldest char, append the new one. keeps context_length fixed at 64
        context = torch.cat([context[:, 1:], torch.tensor([[next_char]])], dim=1)
    print("".join(result))


generate(model)
torch.save(model.state_dict(), "model_weights_2.pth")
