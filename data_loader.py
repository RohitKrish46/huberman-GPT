import torch
from common.config_constants import BLOCK_SIZE, BATCH_SIZE, DEVICE

# Read the input text file
with open('data/Huberman_input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

# Create a list of unique characters in the text and determine vocabulary size
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create dictionaries for character-to-index and index-to-character mapping
ctoi = { ch:i for i, ch in enumerate(chars)} # character-to-index mapping
itoc = { i:ch for i, ch in enumerate(chars)} # index-to-character mapping

# Functions for encoding and decoding text using character mappings
encode = lambda s:[ctoi[c] for c in s]  # encoding  
decode = lambda l:''.join([itoc[i] for i in l])  # decoding   

# Split the data into training and validation sets
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.85*len(data))
train_data = data[:n]
val_data = data[n:]

# Function to generate batch of data for training or validation
def get_batch(split):
    data = train_data if split =='train'else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1 : i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y