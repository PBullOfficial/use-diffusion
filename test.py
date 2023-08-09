import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.__version__)