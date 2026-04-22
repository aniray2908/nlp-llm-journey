# Tensors, Slicing, Broadcasting & GPU Basics

> **Path:** `00_pytorch_warmup/concepts/`  
> **Topic:** PyTorch fundamentals — the building blocks everything else sits on.

---

## Table of Contents

1. [What is a Tensor?](#1-what-is-a-tensor)
2. [Creating Tensors](#2-creating-tensors)
3. [Slicing & Indexing](#3-slicing--indexing)
4. [Useful Tensor Operations](#4-useful-tensor-operations)
5. [Broadcasting](#5-broadcasting)
6. [GPU Basics](#6-gpu-basics)
7. [How This Connects to LLMs](#7-how-this-connects-to-llms)

---

## 1. What is a Tensor?

A tensor is a **generalisation of arrays to any number of dimensions**. Every piece of data in PyTorch — inputs, weights, gradients, outputs — lives as a tensor.

| Structure | Dimensions | Example |
|---|---|---|
| Scalar | 0D | `3.14` |
| Vector | 1D | `[1, 2, 3]` |
| Matrix | 2D | `[[1, 2], [3, 4]]` |
| Stack of matrices | 3D+ | shape `(batch, rows, cols)` |

In the context of LLMs, you'll work with 3D tensors constantly — typically shaped `(batch_size, sequence_length, embedding_dim)`. Every sentence in a batch is a matrix of token vectors, and you stack those matrices into a single 3D tensor.

---

## 2. Creating Tensors

```python
import torch

# From data
scalar = torch.tensor(3.14)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])

# From shape — filled values
zeros  = torch.zeros(3, 4)        # all zeros, shape (3, 4)
ones   = torch.ones(3, 4)         # all ones
eye    = torch.eye(3)             # identity matrix, shape (3, 3)

# From shape — random values
rand   = torch.rand(3, 4)         # uniform random [0, 1)
randn  = torch.randn(3, 4)        # standard normal (mean=0, std=1)

# Sequential
seq    = torch.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
lin    = torch.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]

# Inspecting a tensor
x = torch.randn(3, 4)
print(x.shape)    # torch.Size([3, 4])
print(x.dtype)    # torch.float32
print(x.device)   # device('cpu')
print(x.ndim)     # 2
```

### Data types worth knowing

| dtype | Description | Common use |
|---|---|---|
| `torch.float32` | 32-bit float | Default for model weights |
| `torch.float16` | 16-bit float | Mixed precision training |
| `torch.int64` | 64-bit integer | Token IDs, indices |
| `torch.bool` | Boolean | Attention masks |

```python
# Casting between dtypes
x = torch.tensor([1, 2, 3])            # int64 by default
x = x.float()                           # → float32
x = x.to(torch.float16)                 # → float16
```

---

## 3. Slicing & Indexing

Slicing works just like NumPy. Each comma separates a dimension.

```python
x = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Basic indexing
x[0]          # → tensor([1, 2, 3])      first row
x[0, 2]       # → tensor(3)              row 0, col 2
x[-1]         # → tensor([7, 8, 9])      last row
x[0, -1]      # → tensor(3)              first row, last element

# Range slicing  [start:stop:step]
x[:, 1]       # → tensor([2, 5, 8])      all rows, col 1
x[1:, :2]     # → tensor([[4, 5],        rows 1+, cols 0-1
              #            [7, 8]])
x[::2]        # → rows 0, 2             every other row

# Fancy indexing — pass a list of indices
x[[0, 2]]     # → rows 0 and 2
x[range(3), range(3)]  # → diagonal: tensor([1, 5, 9])
```

### The diagonal trick explained

`x[range(3), range(3)]` passes `[0,1,2]` as row indices and `[0,1,2]` as column indices simultaneously. PyTorch zips them — giving you `(row0,col0)`, `(row1,col1)`, `(row2,col2)` — which is exactly the diagonal. You can verify with `torch.diag(x)`.

### Reshaping

```python
x = torch.randn(3, 4)

x.reshape(2, 6)    # → shape (2, 6) — total elements must match
x.view(2, 6)       # same, but requires contiguous memory
x.flatten()        # → shape (12,) — collapses everything
x.unsqueeze(0)     # → shape (1, 3, 4) — adds a dimension at pos 0
x.squeeze()        # removes all dimensions of size 1
x.T                # transpose — shape (4, 3)
```

> **`reshape` vs `view`:** Prefer `reshape` — it works on non-contiguous tensors by making a copy if needed. `view` is faster but will error if the tensor isn't contiguous in memory (which can happen after certain operations like `.T`).

---

## 4. Useful Tensor Operations

```python
x = torch.randn(3, 4)

# Reduction operations
x.sum()           # sum of all elements → scalar
x.sum(dim=0)      # sum along rows → shape (4,)
x.sum(dim=1)      # sum along cols → shape (3,)
x.mean()          # mean of all elements
x.mean(dim=1)     # mean per row → shape (3,)
x.max()           # single max value
x.max(dim=0)      # max per column, returns (values, indices)
x.argmax(dim=1)   # index of max per row

# Element-wise operations
x + 1             # add scalar to every element
x * 2             # multiply every element
x ** 2            # square every element
torch.sqrt(x.abs()) # sqrt of abs values

# Matrix operations
a = torch.randn(3, 4)
b = torch.randn(4, 5)
torch.matmul(a, b)    # matrix multiply → shape (3, 5)
a @ b                  # shorthand for matmul

# Stacking and concatenation
a = torch.ones(3, 4)
b = torch.ones(3, 4)
torch.cat([a, b], dim=0)    # → shape (6, 4) — stack along rows
torch.cat([a, b], dim=1)    # → shape (3, 8) — stack along cols
torch.stack([a, b], dim=0)  # → shape (2, 3, 4) — new dimension
```

### Understanding `dim` — the mental model

> **The dimension you name is the one that gets collapsed away.**

```python
x = torch.randn(3, 4)  # 3 rows, 4 cols

x.sum(dim=0)  # collapse rows   → one value per column → shape (4,)
x.sum(dim=1)  # collapse cols   → one value per row    → shape (3,)
x.sum()       # collapse all    → scalar
```

This intuition carries directly into Transformer code — e.g. `softmax(dim=-1)` means "normalise across the last dimension", which is exactly what you want when computing attention weights across tokens.

---

## 5. Broadcasting

Broadcasting lets PyTorch do arithmetic between tensors of **different shapes** without manually resizing them. No extra memory is used — it's handled virtually.

### The rule

> Align shapes from the **right**. A dimension is compatible if it either matches, or one of them is `1`. A missing dimension is treated as `1`.

```
shape (3, 4) + shape (4,)
→ right-align: (3, 4) and (1, 4)  ← (4,) treated as (1, 4)
→ broadcast:   (3, 4) and (3, 4)  ← the 1 expands to 3
→ result:      (3, 4) ✅
```

### Examples

```python
# Adding a row vector to every row of a matrix
a = torch.tensor([[1, 2, 3],   # shape (2, 3)
                   [4, 5, 6]])
b = torch.tensor([10, 20, 30]) # shape (3,)

a + b
# → tensor([[11, 22, 33],
#            [14, 25, 36]])
```

```python
# Adding a column vector to every column of a matrix
col = torch.tensor([[1],   # shape (3, 1)
                    [2],
                    [3]])
row = torch.tensor([10, 20, 30])  # shape (3,) → treated as (1, 3)

col + row
# → tensor([[11, 21, 31],   ← outer sum
#            [12, 22, 32],
#            [13, 23, 33]])
```

```python
# The reshape trick — adding per-row bias
matrix   = torch.zeros(4, 3)
row_bias = torch.tensor([10.0, 20.0, 30.0, 40.0])  # shape (4,)

# ❌ won't broadcast correctly — (4,) aligns with last dim (3), mismatch
matrix + row_bias

# ✅ reshape to column vector (4, 1) — now broadcasts across all 3 cols
matrix + row_bias.reshape(4, 1)
# → [[10, 10, 10],
#    [20, 20, 20],
#    [30, 30, 30],
#    [40, 40, 40]]
```

### Broadcasting shape cheatsheet

| Shape A | Shape B | Result | Notes |
|---|---|---|---|
| `(3, 4)` | `(4,)` | `(3, 4)` | B broadcast across rows |
| `(3, 1)` | `(1, 4)` | `(3, 4)` | Both expand |
| `(5, 1, 4)` | `(3, 4)` | `(5, 3, 4)` | B treated as `(1, 3, 4)` |
| `(3, 4)` | `(3,)` | ❌ | Right-align fails: 4 ≠ 3 |

> **Tip:** When unsure, right-align the shapes on paper and check dimension by dimension from right to left. If any pair is neither equal nor 1, it won't broadcast.

---

## 6. GPU Basics

PyTorch tensors live on a **device** — CPU or GPU. Moving computation to a GPU can speed up training by 10–100× for large models.

### Checking and setting the device

```python
# Best practice — detect once, use everywhere
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# On Google Colab: Runtime → Change runtime type → T4 GPU
```

### Moving tensors

```python
x = torch.tensor([1.0, 2.0, 3.0])   # CPU by default
print(x.device)                       # cpu

x = x.to(device)                      # move to GPU (if available)
print(x.device)                       # cuda:0

x = x.to('cpu')                       # move back to CPU

# Alternative shorthand
x = x.cuda()    # → GPU
x = x.cpu()     # → CPU
```

> **Important:** `.to()` returns a **new tensor** — it does not modify in place. Always reassign: `x = x.to(device)`.

### The device mismatch error

```python
a = torch.tensor([1.0]).to('cuda')   # GPU
b = torch.tensor([2.0])              # CPU

a + b
# ❌ RuntimeError: Expected all tensors to be on the same device,
#    but found at least two devices, cuda:0 and cpu!

# Fix: move b to match a
b = b.to(a.device)
a + b  # ✅
```

PyTorch forces you to be explicit about devices — this is intentional. Accidental CPU/GPU mixing would silently destroy training performance.

### The standard training pattern

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model  = model.to(device)
inputs = inputs.to(device)
labels = labels.to(device)
```

You'll write this at the top of every training script. It's boilerplate, but important boilerplate.

### Writing device-agnostic functions

```python
# Approach 1: detect device inside the function
def safe_matmul(a, b):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = a.to(device)
    b = b.to(device)
    return torch.matmul(a, b)

# Approach 2: infer device from input (more flexible)
def safe_matmul(a, b):
    b = b.to(a.device)     # match b to wherever a already lives
    return torch.matmul(a, b)
```

Approach 2 is preferred in real code — it respects whatever device the caller already set up, rather than overriding it.

---

## 7. How This Connects to LLMs

Every concept in this notebook shows up constantly in Transformer and LLM code:

| Concept | Where it appears in LLMs |
|---|---|
| **3D tensors** | `(batch, seq_len, embed_dim)` — the shape of everything |
| **Slicing** | Extracting tokens, applying causal masks, teacher forcing |
| **`dim` in reductions** | `softmax(dim=-1)` for attention weights |
| **Broadcasting** | Adding positional encodings to token embeddings |
| **Column vector reshape** | Bias terms in attention projections |
| **`matmul` / `@`** | Q×Kᵀ in attention, every linear layer |
| **`.to(device)`** | Moving batches to GPU in the training loop |

The bonus exercise from the demos folder is worth revisiting with this in mind:

```python
embeddings = torch.randn(8, 10, 16)  # batch=8, tokens=10, embed_dim=16
pos_enc    = torch.randn(10, 16)     # one encoding per token position

# Broadcasting: pos_enc (10, 16) → treated as (1, 10, 16) → expands to (8, 10, 16)
out = embeddings + pos_enc           # no reshape needed — it just works

# Slice first 5 tokens of each sentence
out[:, :5, :]                        # shape (8, 5, 16)

# Mean across token dimension
out.mean(dim=1)                      # shape (8, 16) — one vector per sentence
```

This is real Transformer input processing. You just did it.

---

*Next concept → [Autograd](./02_autograd.md)*
