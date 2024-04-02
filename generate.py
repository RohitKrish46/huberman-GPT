from gpt import Huberman_GPT, decode, encode
from common.config_constants import DEVICE
import torch

model = Huberman_GPT()
model.load_state_dict(torch.load("models/huberman_model.pth"))
m = model.to(DEVICE)

user_input = input("Enter something: ")

context = torch.tensor(encode(user_input), dtype=torch.long, device=DEVICE)
context = context.view(context.shape[0],1)

print(decode(m.generate(context, max_new_tokens=4000)[0].tolist()))