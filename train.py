import torch
from torch.onnx import TrainingMode

training_data = []
testing_data = []
forward_map = {}
reverse_map = {}
tokenised_text = []
n = 0

# these are the simple functions that are used so i am declaring them here and at the bottom they would be called accordingly to their use case .
# and also the functions can be added this way right ?


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
#
data = torch.tensor(tokenised_text)
split_point = int(0.9 * (len(data)))
training_data = data[:split_point]
testing_data = data[split_point:]
print(len(training_data))
print(len(testing_data))
