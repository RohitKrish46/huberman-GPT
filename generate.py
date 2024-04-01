from gpt import BigramLanguageModel, decode, encode
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'




model = BigramLanguageModel()
model.load_state_dict(torch.load("models/rogan_model.pth"))
m = model.to(device)

# Get user input for context generation
user_input = input("Enter something: ")

# encode the user's input and create a tensor of it
context = torch.tensor(encode(user_input), dtype=torch.long, device=device)
context = context.view(context.shape[0],1)

print(decode(m.generate(context, max_new_tokens=4000)[0].tolist()))