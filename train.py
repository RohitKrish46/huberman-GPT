import torch
from gpt import Huberman_GPT
from data_loader import get_batch
from common.config_constants import DEVICE, LEARNING_RATE, MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS
model = Huberman_GPT()
m = model.to(DEVICE)

optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


for iter in range(MAX_ITERS): 


    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step{iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

