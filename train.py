# references - github.com/avirals554/STM32-inference-engines
# important facts that needs to be remembered for checking the code and doing stuff -
# the size of the dictionary is - 65--forward_map
# the size of the training dataset- 1003854
# the size of the testing dataset-111540

import torch
import torch.nn as nn
from torch.special import softmax

training_data = []
testing_data = []
forward_map = {}
reverse_map = {}
tokenised_text = []
n = 0

# these are the simple functions that are used so i am declaring them here and at the bottom they would be called accordingly to their use case .
# and also the functions can be added this way right ?


# this is a simple function that creates an array of embeddings that are 65 in number, i am not using the input stuff at all cause i am thinking the following-
# does "mapping " actually useful ? are are we just doing stuff that is not needed at all ? i mean we can just create a list of 65 character embeddings
# and then we are just gonna search it through the embeddings so if lets say that we have the id of 1 for 'A' in the dictionary why not just use it
# as a concept in the embedding dictionary too ? without physically mapping it .
# ok so we are gonna create this so that


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # setting the constructor for the initial values that we are every gonna need for the training of the data
        self.char_embedding = nn.Embedding(65, 64)
        self.pos_embedding = nn.Embedding(64, 64)
        self.query = nn.Linear(64, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.mask = torch.tril(torch.ones(64, 64))
        self.ff1 = nn.Linear(64, 128)
        self.ff2 = nn.Linear(128, 64)
        self.output_head = nn.Linear(64, 65)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)

    def forward(self, x):
        # feed forward function
        x = self.char_embedding(x) + self.pos_embedding(torch.arange(64))
        # this is the start of the attention stuff i am writting this as a way to seperate the code in section inside a functions
        #
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        A = (Q @ K.transpose(-2, -1)) / 64**0.5

        A = A.masked_fill(self.mask == 0, float("-inf"))
        At = A.softmax(dim=-1)
        # the -1 this is just to tell the
        output = At @ V
        # this is where the attention ends and we start with the feed forward thing that will give us the predictions
        # added another form of normalization bellow to improve accuracy the first time the loss function reached 1.8 max now after adding the
        # bellow line it reached to like 1.5 something
        x = x + output
        x = self.norm1(x)

        output = self.ff1(x)
        output = torch.relu(output)
        output = self.ff2(output)

        x = x + output  # ← merge back into main flow
        x = self.norm2(x)
        x = self.output_head(x)
        return x


def get_batch():
    # so this is function is simple we are gonna take the training data list and then we are gonna make "batches" of it the idea is that that
    # batches would look something like this --
    # [0.....64] : [1.....65] , [1.....65]:[2......66],[2.......66]:[3.......67]..... and so on .
    input_vector = []
    value_vector = []
    # so this was interesting right cause in this we are choosing random numbers or indices from the list and then we are gonna use that positions
    # and have a data set of 64 FROM that index and then make the collection of those , and random thing is important so that the llm doesnt just
    # remembers that patter of the txt itself but actually learn the meaning and how to form sentences .
    positions = torch.randint(len(training_data) - 64, (32,))
    for i in positions:
        input_vector.append(training_data[i : i + 64])
        value_vector.append(training_data[i + 1 : (i + 64) + 1])
        # the reason that we are adding stack method here is cause we made the arrays of stuff in the previous step and then we want to join those arrays as well
        # this got a bit confusing but essentially what is haapening is this --
        # we dont declare if the vector is 2d or 3d hence i am guessing it can become 2d its not like vector<vector<int>> so that would mean that
        # you need to joing the 1D vectors to make a 2D vector and in order to do that we use .stack() method
    batch_input = torch.stack(input_vector)
    batch_value = torch.stack(value_vector)
    return batch_input, batch_value


def mapping():
    # this is gonna take the txt file and iterate it one character at a time and then make a dictionary that we are gonna call forward_map()
    # and this is gonna contain all character that are in this .txt file
    # the ids that are given are just simple integers that are incremented by one for each new character , it checks whether that character has appeared before and
    # if not then it adds it to the dictionary
    # Example-- 'A'-->0,'B'-->1,'C'-->2......

    global n
    with open("input.txt", "r") as i:
        while True:
            char = i.read(1)
            if not char:
                break
            if char not in forward_map:
                forward_map[char] = n
                n += 1
            else:
                continue


def reverse_mapping():
    # this is for the revere mapping of the mapping function this function just reverces the dictionary and makes it so that that
    # the keys become the value and the values become the keys so its like this --
    # Example - 0-->'A',1-->'B',2-->'C'.......
    global reverse_map
    reverse_map = {v: k for k, v in forward_map.items()}


def tokenise_local():
    global tokenised_text
    with open("input.txt", "r") as i:
        while True:
            char = i.read(1)
            if not char:
                break
            tokenised_text.append(forward_map[char])


# the following 3 functions are just for the pretraining of the data and nothing more but they should be here as to not get confused or change their order
mapping()
reverse_mapping()
tokenise_local()
# after the data is ready we are gonna use the torch library finally and then print the data
# this step is important cause now we can actually do stuff with the data , in python to perform any action we must iterate the entire list and then do stuff with it but pytorch is
# is just annoyingly useful in this case .
print(forward_map)
data = torch.tensor(tokenised_text)
split_point = int(0.9 * (len(data)))
training_data = data[:split_point]
testing_data = data[split_point:]

# this is the part for the declaration of the model or the tiny transformer
model = TinyTransformer()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for step in range(50000):
    # calling the function that we made for the getting the set of the inputs and the outputs
    inputs, targets = get_batch()

    predictions = model(inputs)
    predictions = predictions.view(-1, 65)
    targets = targets.view(-1)
    loss = loss_fn(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"step {step}, loss: {loss.item():.4f}")


def generate(model, max_chars=200):
    context = torch.zeros(1, 64, dtype=torch.long)
    result = []
    for i in range(max_chars):
        output = model(context)
        probs = torch.softmax(output[0, -1], dim=0)
        next_char = torch.multinomial(probs, 1).item()
        result.append(reverse_map[next_char])
        context = torch.cat([context[:, 1:], torch.tensor([[next_char]])], dim=1)
    print("".join(result))


generate(model)
torch.save(model.state_dict(), "model_weights_1.pth")
