# nn.Module & Custom Layers

> **Path:** `00_pytorch_warmup/concepts/`  
> **Topic:** PyTorch's base class for every neural network — how to build, stack, and train custom modules.

---

## Table of Contents

1. [What is `nn.Module`?](#1-what-is-nnmodule)
2. [Anatomy of a Module](#2-anatomy-of-a-module)
3. [Parameters vs Buffers](#3-parameters-vs-buffers)
4. [Activation Functions](#4-activation-functions)
5. [Stacking Layers](#5-stacking-layers)
6. [`nn.Sequential` — Shorthand for Simple Stacks](#6-nnsequential----shorthand-for-simple-stacks)
7. [Nested Modules](#7-nested-modules)
8. [Useful Built-in Methods](#8-useful-built-in-methods)
9. [`train()` vs `eval()` Mode](#9-train-vs-eval-mode)
10. [Shape Flow — The Most Important Debugging Skill](#10-shape-flow----the-most-important-debugging-skill)
11. [How This Connects to LLMs](#11-how-this-connects-to-llms)

---

## 1. What is `nn.Module`?

`nn.Module` is PyTorch's **base class for every neural network**. Any model you build — a single layer, a Transformer block, a full GPT — is a subclass of `nn.Module`.

It gives you three things automatically:

- **Parameter tracking** — any `nn.Parameter` or sub-module you register is automatically collected by `.parameters()`, which the optimiser uses to update weights
- **Forward pass structure** — you define `forward()` and PyTorch handles calling it correctly with all necessary hooks
- **Utility methods** — `.to(device)`, `.train()`, `.eval()`, `.state_dict()`, `.load_state_dict()` all work out of the box on the entire model tree

---

## 2. Anatomy of a Module

Every module has exactly two methods you need to define:

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()      # always call this first — sets up internal PyTorch state
        # define layers and parameters here

    def forward(self, x):
        # define the computation — what happens to input x
        return x
```

### Calling the model

```python
model = MyModel()

# ✅ correct — call the model like a function
out = model(x)

# ❌ avoid — bypasses PyTorch's internal hooks
out = model.forward(x)
```

Using `model(x)` invokes `__call__`, which wraps `forward()` with important hooks for things like gradient tracking and registered hooks. Always use the function-call syntax.

### A concrete example — linear layer from scratch

```python
class SimpleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias

model = SimpleLinear(3, 2)
x     = torch.randn(5, 3)
out   = model(x)
print(out.shape)    # torch.Size([5, 2])
```

This is exactly what `nn.Linear` does under the hood.

---

## 3. Parameters vs Buffers

Inside a module, there are two kinds of tensors:

### Parameters — learned during training

Registered with `nn.Parameter`. Automatically included in `.parameters()` and updated by the optimiser:

```python
self.weight = nn.Parameter(torch.randn(out_f, in_f))
self.bias   = nn.Parameter(torch.zeros(out_f))
```

Any `nn.Linear`, `nn.Embedding`, `nn.LayerNorm` etc. you define as attributes are also automatically registered — PyTorch walks the module tree and finds them.

### Buffers — part of the model but not learned

Registered with `self.register_buffer()`. Saved in `state_dict()` and moved with `.to(device)`, but not updated by the optimiser:

```python
class ModelWithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        # e.g. a fixed positional encoding or running mean
        self.register_buffer('pos_enc', torch.randn(10, 4))

    def forward(self, x):
        return self.linear(x) + self.pos_enc[:x.size(0)]
```

Common buffer use cases: positional encodings, running statistics in batch norm, attention masks that don't change.

### Checking what's registered

```python
# Parameters
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)

# Buffers
for name, buf in model.named_buffers():
    print(name, buf.shape)
```

---

## 4. Activation Functions

Activation functions introduce **non-linearity** — without them, stacking linear layers is mathematically equivalent to a single linear layer, no matter how deep the network.

### Implementing from scratch

```python
class MyReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0)

# Equivalent implementations
class MyReLU(nn.Module):
    def forward(self, x):
        return torch.where(x > 0, x, torch.zeros_like(x))

class MyReLU(nn.Module):
    def forward(self, x):
        return x * (x > 0)    # boolean mask — elegant one-liner
```

The boolean mask version — `x * (x > 0)` — is worth understanding deeply. `(x > 0)` produces a boolean tensor which acts as `[0, 0, 1, 1, ...]` when multiplied. You will see this exact pattern again in **attention masking**, where future tokens are zeroed out using a boolean mask.

### Common activations

| Activation | Formula | Common use |
|---|---|---|
| `nn.ReLU()` | `max(0, x)` | Hidden layers in MLPs |
| `nn.GELU()` | smooth approximation of ReLU | Transformers, BERT, GPT |
| `nn.Sigmoid()` | `1 / (1 + e^-x)` | Binary classification output |
| `nn.Tanh()` | `(e^x - e^-x) / (e^x + e^-x)` | RNNs, normalising outputs |
| `nn.Softmax(dim=-1)` | normalises to probabilities | Final layer for classification |

> **Note:** GPT and most modern LLMs use **GELU** rather than ReLU — it's smoother around zero which helps with gradient flow in very deep networks.

---

## 5. Stacking Layers

```python
class TwoLayerNet(nn.Module):
    def __init__(self, in_f, hidden_f, out_f):
        super().__init__()
        self.layer1 = nn.Linear(in_f, hidden_f)     # in_f  → hidden_f
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(hidden_f, out_f)    # hidden_f → out_f ← not in_f!

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = TwoLayerNet(4, 16, 1)
x = torch.randn(8, 4)
print(model(x).shape)    # torch.Size([8, 1])
```

### Parameter count — working it out by hand

For `TwoLayerNet(4, 16, 1)`:

| Layer | Weights | Bias | Total |
|---|---|---|---|
| `layer1` | 4 × 16 = 64 | 16 | 80 |
| `layer2` | 16 × 1 = 16 | 1  | 17 |
| **Total** | | | **97** |

```python
total = sum(p.numel() for p in model.parameters())
print(total)    # 97
```

Always estimate parameter counts by hand before building — it's a core skill for designing LLM architectures where parameter budgets matter enormously.

---

## 6. `nn.Sequential` — Shorthand for Simple Stacks

When layers feed one into the next with no branching, `nn.Sequential` saves boilerplate:

```python
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

x = torch.randn(8, 4)
print(model(x).shape)    # torch.Size([8, 1])
```

Mathematically identical to `TwoLayerNet` above.

### When to use which

| Use `nn.Sequential` | Use full `nn.Module` |
|---|---|
| Layers feed straight through | Skip/residual connections |
| No branching | Multiple inputs or outputs |
| Quick prototyping | Custom forward logic |
| Simple feature extractors | Transformer blocks, attention |

In LLM code you'll almost always write full `nn.Module` subclasses — Transformer blocks have residual connections and layer norm that can't be expressed as a simple sequential stack.

---

## 7. Nested Modules

Modules can contain other modules arbitrarily deep. PyTorch tracks the entire tree automatically:

```python
class Block(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.linear1 = nn.Linear(features, features)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(features, features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = Block(8)
        self.block2 = Block(8)
        self.block3 = Block(8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output(x)
        return x


model = Network()
x = torch.randn(4, 8)
print(model(x).shape)    # torch.Size([4, 1])

# Named parameters show the full tree path
for name, param in model.named_parameters():
    print(name, param.shape)
# block1.linear1.weight  torch.Size([8, 8])
# block1.linear1.bias    torch.Size([8])
# block1.linear2.weight  torch.Size([8, 8])
# ...
# output.weight          torch.Size([1, 8])
# output.bias            torch.Size([1])
```

This nesting pattern is exactly how a Transformer is structured — `GPT → TransformerBlock × N → MultiHeadAttention → Linear projections`.

### Using `nn.ModuleList` for dynamic stacking

When you want to stack a variable number of identical blocks:

```python
class DeepNetwork(nn.Module):
    def __init__(self, features, num_layers):
        super().__init__()
        # nn.ModuleList — registers each block as a proper submodule
        self.blocks = nn.ModuleList([Block(features) for _ in range(num_layers)])
        self.output = nn.Linear(features, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.output(x)

model = DeepNetwork(features=8, num_layers=6)
```

> **Important:** Use `nn.ModuleList` not a plain Python list. A plain list won't register the sub-modules, so their parameters won't be found by `.parameters()` and won't be updated during training.

---

## 8. Useful Built-in Methods

```python
model = TwoLayerNet(4, 16, 1)

# Inspect parameters
for name, param in model.named_parameters():
    print(f"{name:20s} shape={param.shape} requires_grad={param.requires_grad}")

# Count parameters
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total} | Trainable: {trainable}")

# Move entire model to device (moves all parameters and buffers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = model.to(device)

# Save model weights
torch.save(model.state_dict(), 'model.pt')

# Load model weights
model.load_state_dict(torch.load('model.pt'))

# Print model architecture summary
print(model)
# TwoLayerNet(
#   (layer1): Linear(in_features=4, out_features=16, bias=True)
#   (relu): ReLU()
#   (layer2): Linear(in_features=16, out_features=1, bias=True)
# )
```

---

## 9. `train()` vs `eval()` Mode

Some layers behave differently during training vs inference. The two most important are:

**Dropout** — randomly zeroes activations during training to prevent overfitting. During eval, all neurons are active and outputs are scaled to compensate.

**Batch Normalisation** — uses batch statistics during training, but running (accumulated) statistics during eval.

```python
model.train()    # training mode — dropout active, batchnorm uses batch stats
model.eval()     # eval mode    — dropout disabled, batchnorm uses running stats

print(model.training)    # True or False
```

### Why this matters for LLMs

During **text generation**, you always want `model.eval()` + `torch.no_grad()`. Forgetting `model.eval()` means dropout randomly zeroes parts of the network mid-generation — producing different (worse) outputs every time you call the model with the same input. This is a subtle bug that can be hard to track down.

```python
# Always use both together for inference
model.eval()
with torch.no_grad():
    output = model(input_ids)
```

---

## 10. Shape Flow — The Most Important Debugging Skill

The most common error when building custom modules is a shape mismatch — `mat1 and mat2 shapes cannot be multiplied`. The fix is always to trace the shape through every layer:

```python
# Always trace shapes when debugging
x = torch.randn(8, 4)
print(f"Input:  {x.shape}")          # (8, 4)

x = model.layer1(x)
print(f"After layer1: {x.shape}")    # (8, 16)

x = model.relu(x)
print(f"After relu:   {x.shape}")    # (8, 16) — relu doesn't change shape

x = model.layer2(x)
print(f"After layer2: {x.shape}")    # (8, 1)
```

### Common shape mistakes

| Mistake | Error message | Fix |
|---|---|---|
| `layer2 = nn.Linear(in_f, out_f)` instead of `nn.Linear(hidden_f, out_f)` | `mat1 and mat2 shapes cannot be multiplied` | Use `hidden_f` as input to layer2 |
| Stale variable from earlier cell | Wrong tensor passed to model | Restart kernel, use specific variable names |
| Forgetting `.unsqueeze()` on 1D input | Shape mismatch | Add batch dimension: `x.unsqueeze(0)` |
| Using plain Python list instead of `nn.ModuleList` | Parameters not updated | Replace with `nn.ModuleList` |

> **Debugging tip:** When a shape error appears, add a `print(x.shape)` after every single operation in `forward()` until you find where it breaks. This is faster than staring at the error.

---

## 11. How This Connects to LLMs

Every LLM is a deeply nested `nn.Module` tree:

```
GPT (nn.Module)
├── token_embedding     (nn.Embedding)
├── position_embedding  (nn.Embedding or nn.Parameter)
├── dropout             (nn.Dropout)
└── blocks              (nn.ModuleList of TransformerBlock)
    └── TransformerBlock (nn.Module) × N
        ├── ln_1            (nn.LayerNorm)
        ├── attn            (nn.Module — MultiHeadAttention)
        │   ├── c_attn      (nn.Linear — projects to Q, K, V)
        │   └── c_proj      (nn.Linear — output projection)
        ├── ln_2            (nn.LayerNorm)
        └── mlp             (nn.Module — FeedForward)
            ├── c_fc        (nn.Linear)
            ├── gelu        (nn.GELU)
            └── c_proj      (nn.Linear)
```

When you call `model.parameters()` on a GPT, PyTorch walks this entire tree and collects every single learnable tensor automatically — that's `nn.Module`'s composability paying off.

| Concept | Where it appears in LLMs |
|---|---|
| `nn.Parameter` | Embedding tables, learned positional encodings |
| `register_buffer` | Fixed positional encodings, attention masks |
| `nn.ModuleList` | Stacking N identical Transformer blocks |
| `nn.Linear` | Q, K, V projections, feed-forward layers |
| `nn.LayerNorm` | Before attention and feed-forward in each block |
| `nn.GELU` | Activation inside every feed-forward block |
| `model.eval()` | Every inference and generation call |
| `named_parameters()` | LoRA — selectively freezing and unfreezing layers |
| Shape flow | Debugging `(batch, seq_len, embed_dim)` mismatches |

---

*Previous concept → [02 — Autograd](./02_autograd.md)*  
*Next concept → [04 — DataLoader, Training Loop & Optimisers](./04_dataloader_training_loop_optimisers.md)*
