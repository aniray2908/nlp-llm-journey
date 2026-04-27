# Autograd — How Gradients are Tracked and Computed

> **Path:** `00_pytorch_warmup/concepts/`  
> **Topic:** PyTorch's automatic differentiation engine — the magic behind neural network training.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [The Computation Graph](#2-the-computation-graph)
3. [`.backward()` — Computing Gradients](#3-backward----computing-gradients)
4. [The Chain Rule — What's Actually Happening](#4-the-chain-rule----whats-actually-happening)
5. [`requires_grad` — What Gets Tracked](#5-requires_grad----what-gets-tracked)
6. [`torch.no_grad()` — Turning Off Tracking](#6-torchno_grad----turning-off-tracking)
7. [`.detach()` — Cutting a Tensor from the Graph](#7-detach----cutting-a-tensor-from-the-graph)
8. [Gradient Accumulation — The Most Common Gotcha](#8-gradient-accumulation----the-most-common-gotcha)
9. [The Training Loop — Putting It All Together](#9-the-training-loop----putting-it-all-together)
10. [How This Connects to LLMs](#10-how-this-connects-to-llms)

---

## 1. The Big Picture

When you train a neural network, you need to compute the **gradient of the loss with respect to every parameter** — so you know which direction to nudge each weight to reduce the loss. Doing this by hand for a deep network with millions of parameters would be impossible.

PyTorch's **autograd** engine does this automatically. Every operation you perform on a tensor is silently recorded, and when you call `.backward()`, PyTorch walks back through that recording and computes all the gradients for you.

### The Cake Analogy

Think of it this way:

- The **network** is a recipe
- The **weights** are the ingredients (amount of salt, sugar, flour)
- The **loss** is how bad the cake tasted — how wrong the network was
- The **gradient** tells you, for each ingredient, *"if I add a little more of this, does the cake get better or worse, and by how much?"*
- **Backpropagation** is reading the recipe backwards to figure out which ingredient was most responsible for the bad taste
- **`zero_grad()`** is washing the bowl before baking a new cake — without it, old residue contaminates the new batch

Training is just repeating this loop — bake, taste, adjust ingredients, wash bowl, repeat — until the cake is good.

---

## 2. The Computation Graph

When you perform operations on tensors that have `requires_grad=True`, PyTorch silently builds a **dynamic computation graph** — a record of every operation and how tensors connect to each other.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2          # y = x²
z = y + 3           # z = x² + 3

print(z)
# tensor(7., grad_fn=<AddBackward0>)
```

Notice `grad_fn=<AddBackward0>` — PyTorch is telling you this tensor was produced by an addition, and it knows how to differentiate through it.

### Inspecting the graph

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y * 4 + 1

print(z.grad_fn)                    # <AddBackward0>
print(z.grad_fn.next_functions)     # shows the chain leading back to x
```

Each `grad_fn` points to the previous operation in the chain — you can trace the entire graph backwards from the final output to the original input.

### Dynamic vs Static graphs

PyTorch builds the graph **dynamically** — it's recreated fresh every forward pass. This means you can use regular Python control flow (if statements, loops) inside your network, and the graph will correctly reflect whatever path the data took. This is one of the reasons PyTorch became the dominant research framework.

---

## 3. `.backward()` — Computing Gradients

Calling `.backward()` on a scalar tensor triggers **backpropagation** through the entire computation graph:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1     # y = x² + 3x + 1
                             # dy/dx = 2x + 3

y.backward()                 # compute gradients
print(x.grad)                # tensor(7.) — because 2(2) + 3 = 7 ✅
```

The gradient is stored in `x.grad`. PyTorch computed `dy/dx` analytically using the chain rule, automatically.

### Why must the output be a scalar?

`.backward()` requires a scalar (0D tensor) because gradients are defined as *"how much does the output change per unit change in input"* — this only makes sense when the output is a single number. If the output is a vector or matrix, you'd need to specify which output you're differentiating, which requires a `gradient` argument.

In practice, your loss function always returns a scalar, so this is never an issue in real training.

---

## 4. The Chain Rule — What's Actually Happening

For chained operations, PyTorch applies the **chain rule** automatically:

> If `A → B → C`, then `dC/dA = dC/dB × dB/dA`

```python
x = torch.tensor(3.0, requires_grad=True)
a = x * 2       # a = 2x          →  da/dx = 2
b = a ** 2      # b = a² = 4x²   →  db/da = 2a = 2(2x) = 4x
c = b + 1       # c = 4x² + 1    →  dc/db = 1

# Chain rule: dc/dx = dc/db × db/da × da/dx
#                   = 1 × 4x × 2
#                   = 8x
#                   = 8(3) = 24

c.backward()
print(x.grad)   # tensor(24.) ✅
```

In a deep neural network with 96 layers (like GPT-3), the chain rule is applied through every single layer automatically. This is the core of why deep learning is computationally feasible.

---

## 5. `requires_grad` — What Gets Tracked

By default, tensors don't track gradients. You opt in explicitly:

```python
x = torch.tensor(2.0)                       # not tracked
x = torch.tensor(2.0, requires_grad=True)   # tracked

# Check
print(x.requires_grad)    # True
```

### Propagation through operations

If any input to an operation has `requires_grad=True`, the output will too:

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0)                        # no grad tracking

c = a * b
print(c.requires_grad)    # True  — because a is tracked
print(c.grad_fn)          # <MulBackward0>

print(b.requires_grad)    # False — b itself is not tracked
print(b.grad_fn)          # None
```

### Model parameters are always tracked

```python
import torch.nn as nn

layer = nn.Linear(3, 2)
print(layer.weight.requires_grad)   # True — always, automatically
print(layer.bias.requires_grad)     # True
```

You never need to set `requires_grad=True` on model parameters manually — PyTorch does this for you when you define a layer.

---

## 6. `torch.no_grad()` — Turning Off Tracking

During **inference** (making predictions, not training), you don't need gradients. Wrapping code in `torch.no_grad()` disables graph construction entirely — saving memory and speeding things up:

```python
x = torch.tensor(2.0, requires_grad=True)

# Without no_grad — graph is built
y = x ** 2
print(y.requires_grad)    # True

# With no_grad — no graph, no memory overhead
with torch.no_grad():
    z = x ** 2
print(z.requires_grad)    # False
```

### Standard inference pattern

```python
model.eval()
with torch.no_grad():
    predictions = model(inputs)
```

You'll write this in every evaluation loop, every generation loop, and any time you're using a model without updating its weights.

---

## 7. `.detach()` — Cutting a Tensor from the Graph

`.detach()` creates a new tensor with the same values but **severed from the computation graph**:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2               # y is part of the graph

y_detached = y.detach()  # same value, no connection to graph
print(y_detached)                  # tensor(4.)
print(y_detached.requires_grad)    # False
print(y_detached.grad_fn)          # None
```

### `detach()` vs `no_grad()` — what's the difference?

| | `no_grad()` | `detach()` |
|---|---|---|
| **Scope** | Block of code | Single tensor |
| **Use case** | Inference, evaluation | Extracting a value mid-graph |
| **Original tensor** | Unaffected | Unaffected |
| **Result** | Operations inside block aren't tracked | New tensor with no graph connection |

Common use cases for `.detach()`:
- Logging or plotting loss values mid-training without keeping the graph in memory
- Stopping gradients from flowing into part of a network (e.g. freezing an encoder)
- The reward model in RLHF — you detach the reward signal so it doesn't backprop into the wrong model

---

## 8. Gradient Accumulation — The Most Common Gotcha

Gradients **accumulate** in `.grad` by default — they add on top of whatever was already there. They don't reset automatically between backward passes:

```python
x = torch.tensor(2.0, requires_grad=True)

for i in range(4):
    y = x ** 2
    y.backward()
    print(f"Step {i}: x.grad = {x.grad}")

# Step 0: x.grad = tensor(4.)
# Step 1: x.grad = tensor(8.)   ← accumulating!
# Step 2: x.grad = tensor(12.)
# Step 3: x.grad = tensor(16.)
```

### The fix — zero gradients before each backward pass

```python
x = torch.tensor(2.0, requires_grad=True)

for i in range(4):
    y = x ** 2
    y.backward()
    print(f"Step {i}: x.grad = {x.grad}")
    x.grad.zero_()              # ← reset after each step

# Step 0: x.grad = tensor(4.)
# Step 1: x.grad = tensor(4.)  ✅ correct every time
# Step 2: x.grad = tensor(4.)
# Step 3: x.grad = tensor(4.)
```

### Why `x.grad.zero_()` and not `x.grad = 0`?

The trailing underscore `_` means **in-place operation** in PyTorch. It modifies the existing tensor rather than creating a new one. Setting `x.grad = 0` would replace the grad tensor with a plain Python integer, which breaks PyTorch's internal tracking.

In a real training loop you use `optimizer.zero_grad()` — which calls `.zero_()` on every parameter's gradient at once. Same mechanism, just applied to all parameters simultaneously.

---

## 9. The Training Loop — Putting It All Together

Every neural network training loop follows the same five-step heartbeat:

```python
import torch
import torch.nn as nn

# Data — learning y = 3x + 1
x_train = torch.linspace(0, 1, 10).unsqueeze(1)   # shape (10, 1)
y_train = 3 * x_train + 1

# Model, optimiser, loss
model     = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn   = nn.MSELoss()

for epoch in range(200):
    optimizer.zero_grad()           # 1. wash the bowl — clear old gradients
    pred = model(x_train)           # 2. forward pass — bake the cake
    loss = loss_fn(pred, y_train)   # 3. compute loss — taste the cake
    loss.backward()                 # 4. backprop — who's to blame?
    optimizer.step()                # 5. update weights — adjust ingredients

    if epoch % 40 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

print(f"\nLearned weight: {model.weight.item():.4f}")   # ≈ 3.0
print(f"Learned bias:   {model.bias.item():.4f}")       # ≈ 1.0
```

### Why this order matters

| Step | Why this order |
|---|---|
| `zero_grad()` first | Old gradients must be cleared before computing new ones |
| Forward before loss | You need predictions before you can measure error |
| Loss before backward | `.backward()` needs the loss scalar to differentiate |
| Backward before step | Gradients must exist before the optimiser can use them |

Swapping any of these steps breaks training in subtle, hard-to-debug ways.

---

## 10. How This Connects to LLMs

| Concept | Where it appears in LLMs |
|---|---|
| `requires_grad=True` | All model parameters — weights, biases, embedding tables |
| Computation graph | Built fresh every forward pass through all N Transformer layers |
| Chain rule | Gradients flowing backwards through 96 layers in GPT-3 |
| `.backward()` | Called on cross-entropy loss after every batch |
| `optimizer.zero_grad()` | First line of every training iteration — always |
| `torch.no_grad()` | All inference and text generation — every time you prompt a model |
| `.detach()` | Logging loss curves; stopping gradient flow in frozen layers; RLHF reward signal |
| Gradient accumulation | Intentionally *not* zeroing for several steps to simulate larger batch sizes on limited GPU memory — a common LLM training trick |

### A note on gradient accumulation as a feature

In the gotcha section, accumulation was a bug. But in LLM training it's also used intentionally — if your GPU can only fit a batch size of 4 but you want to train with an effective batch size of 32, you run 8 forward/backward passes without zeroing gradients, then call `optimizer.step()` once. The gradients accumulate across those 8 passes, simulating a larger batch. Same mechanism, intentional use.

---

*Previous concept → [01 — Tensors, Slicing, Broadcasting & GPU](./01_tensors_slicing_broadcasting_gpu.md)*  
*Next concept → [03 — nn.Module & Custom Layers](./03_nn_module_custom_layers.md)*
