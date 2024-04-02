# huberman-GPT
This repository contains the implementation of a character-level transformer model for text generation, inspired by Andrew Huberman's podcast. The model generates text by learning the patterns and structure of podcast text data at the character level. The model learns to predict the next character in a sequence based on the previous characters, capturing long-range dependencies in the text data.


## Dataset
The dataset comprises over 100 word files, each containing transcripts of Dr. Andrew Huberman's podcast episodes.

Download it from [here](https://drive.google.com/drive/folders/1nBfN4QvKHaApCuCGlKg-a4BJm1Ix5k-L)

## Huberman-GPT
- [gpt.py](./gpt.py) : the GPT model
- [data_loader.py](./data_loader.py) : prepares the data for training
- [train.py](./train.py) : training script with optimizer, training loop, model saving etc.
- [generate.py](./generate.py) : generate text after user's input by loading the saved model


  
## Training:
```
python train.py
```

## Text Generation:
```
python generate.py
```
  
## Hyperparameters and Training Results
  ```
batch_size = 64 
block_size = 420 
max_iters = 9000
eval_interval = 500
learning_rate = 5e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

Train loss 0.8142
Val loss 0.9252
```
## Sample Text Generation

```
 So basically the scientists, most people's brain doesn't work, mostly treatments, most ridgentists like me.
 They're at making severe 90%, and many against all the scientists of caffeine, which have varieus vasodilation and in general directing and testosterone.
 We will talk about the behaviors that are at something.
 Now you're in this case that testosterone levels can be done her roughly in a direction between that loss, which brings down a so-called unhealthy effect.
 Now the National Institutes are really occuting studied, focus, and lift of course in the future literature circuits that we call effect, obviously during the skin in enduration and effort.
 But the first time will start to create risk and that will tend to go activation and protect its effects, but you can go away.
 Now that's actually occurring in the muscles from this various kinds of muscle.
 Being into that skin root and will allow summing and just really safely speak large.
 It's really a second.
```
## Credits
Big thanks to [Andrej Karpathy](https://twitter.com/karpathy) or the exceptional [tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&t=1s) on transformers and character-level GPT modeling.

Huge appreciation to [Dr. Andrew Huberman](https://twitter.com/hubermanlab) for consistently entertaining with outstanding podcast on understanding ourselves.
