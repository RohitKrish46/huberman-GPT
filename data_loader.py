import torch
from common.config_constants import BLOCK_SIZE, BATCH_SIZE, DEVICE

with open('data/Huberman_input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from charecters to integers
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s:[stoi[c] for c in s]  # noqa: E731
decode = lambda l:''.join([itos[i] for i in l])  # noqa: E731, E741

# Train and Test splits
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.85*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs Xs and target Ys
    data = train_data if split =='train'else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1 : i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y