# huberman-GPT
Welcome to huberman-GPT, a character-level GPT model trained to generate text inspired by the style, tone, and subject matter of Dr. Andrew Huberman’s neuroscience podcast. This project explores the power of transformers at the character level, focusing on long-range dependencies and text generation fidelity.

## 🧬 Project Overview
This project is a deep learning implementation of a character-level transformer trained on real podcast transcripts from the Huberman Lab. It uses a GPT-style architecture to generate coherent and stylistically consistent text one character at a time.

Inspired by Andrej Karpathy's work, the model captures intricate dependencies across long sequences without tokenizing words, learning directly from raw characters.

📚 Dataset
📄 Over 100+ podcast transcripts from Dr. Andrew Huberman’s podcast Download it from [here](https://drive.google.com/drive/folders/1nBfN4QvKHaApCuCGlKg-a4BJm1Ix5k-L).

Stored as .docx files

Preprocessed into plain text format

Split into:

- 85% training

- 15% validation

## 🧠 Model Architecture

| Component  | Value |
| ------------- | ------------- |
| Embedding Dimension  | `384`  |
|  Attention Heads  | `6`  |
| Transformer Layers  | `6`  |
| Dropout Ratio  | `0.2` |
| Sequence Length (Block Size)  | `420`  |


The model uses:

- ✅ Positional embeddings

- ✅ Multi-head self-attention

- ✅ Layer normalization

- ✅ Feed-forward neural network

- ✅ Causal masking for autoregression


## ⚙️ Training Configuration

| Hyperparameter  | Value |
| ------------- | ------------- |
| Batch Size  | `64`  |
| Learning Rate  | `5e-4`  |
| Max Iterations  | `9000`  |
| Evaluation Interval  | `500` |
| Evaluation Iterations  | `200` |
| Optimizer  | `AdamW` |

## 🔥 Final Results
| Metric  | Value |
| ------------- | ------------- |
| Train Loss  | `0.8142`  |
| Val Loss  | `0.9252`  |

## 🚀 Usage

1. 🔧 Setup
```
git clone https://github.com/Rohitkrish46/huberman-GPT.git
cd huberman-GPT
pip install -r requirements.txt
```
2. 🏋️‍♂️ Train the Model
```
python train.py
```
3. 📝 Generate Text
```
python generate.py
```
   

## 🧪 Sample Generated Text

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
While the model isn't perfect, it does produce structurally and semantically intriguing outputs in the voice and themes of the original podcast content.


## 💡 Inspiration
- 🧑‍🏫 Andrej Karpathy’s nanoGPT — for demystifying transformer implementations

- 🎧 Dr. Andrew Huberman — for consistently producing neuroscience content that blends rigor with accessibility

## 🙌 Acknowledgements
- [Andrej Karpathy](https://twitter.com/karpathy) for the GPT inspiration

- [Dr. Andrew Huberman](https://twitter.com/hubermanlab) for his educational work in neuroscience
