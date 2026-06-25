import torch

state = torch.load("model_weights_2.pth", map_location="cpu")


print(state["query.weight"])
