from gpt import Huberman_GPT, decode, encode
from common.config_constants import DEVICE
import torch

# Initialize the GPT model 
model = Huberman_GPT()

# Load pre-trained weights
model.load_state_dict(torch.load("models/huberman_model.pth"))
m = model.to(DEVICE)

# Get user input
user_input = input("Enter something: ")

# Encode user input into tensor of indices
context = torch.tensor(encode(user_input), dtype=torch.long, device=DEVICE)
context = context.view(context.shape[0],1)

# Print the generated text continuation
print(decode(m.generate(context, max_new_tokens=4000)[0].tolist()))