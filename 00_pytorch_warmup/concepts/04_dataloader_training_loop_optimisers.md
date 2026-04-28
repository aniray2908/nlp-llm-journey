# DataLoader, Training Loop & Optimisers

> **Path:** `00_pytorch_warmup/concepts/`  
> **Topic:** How data is served to a model, how weights are updated, and which optimisers to use when.

---

## Table of Contents

1. [The Problem DataLoader Solves](#1-the-problem-dataloader-solves)
2. [Dataset — Defining Your Data](#2-dataset----defining-your-data)
3. [DataLoader — Serving Batches](#3-dataloader----serving-batches)
4. [Train vs Validation Split](#4-train-vs-validation-split)
5. [Loss Functions](#5-loss-functions)
6. [Optimisers](#6-optimisers)
7. [Learning Rate & Scheduling](#7-learning-rate--scheduling)
8. [The Complete Training Loop](#8-the-complete-training-loop)
9. [Reading Your Training Curves](#9-reading-your-training-curves)
10. [How This Connects to LLMs](#10-how-this-connects-to-llms)

---

## 1. The Problem DataLoader Solves

So far we've fed entire datasets into the model at once. In reality this breaks down fast:

- A dataset of a billion tokens won't fit in GPU memory
- Training on the full dataset before updating weights is slow and unstable
- The model needs to see varied examples in varied order to generalise

A **DataLoader** solves all three by:

- Splitting data into **batches** — small chunks the GPU can handle
- **Shuffling** each epoch so the model doesn't memorise order
- Loading data **in parallel** on CPU so the GPU is never waiting

---

## 2. Dataset — Defining Your Data

`Dataset` is an abstract base class. You subclass it and implement two methods:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)              # total number of samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] # fetch one sample by index
```

PyTorch calls `__len__` to know how many samples exist, and `__getitem__` to fetch individual samples by index. The DataLoader handles everything else — batching, shuffling, parallel loading.

### `TensorDataset` — shortcut for tensor data

When your data is already in tensors, skip the custom class:

```python
from torch.utils.data import TensorDataset

dataset = TensorDataset(X, y)   # wraps tensors directly
```

---

## 3. DataLoader — Serving Batches

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,      # samples per batch
    shuffle=True,       # randomise order each epoch
    num_workers=2,      # CPU threads for parallel loading
    pin_memory=True,    # faster CPU→GPU transfer
    drop_last=False,    # whether to drop the final incomplete batch
)

# Iterating
for X_batch, y_batch in dataloader:
    print(X_batch.shape)   # torch.Size([32, ...])
    break
```

### Key arguments

| Argument | What it does | Typical value |
|---|---|---|
| `batch_size` | Samples per batch | 32, 64, 128, 256 |
| `shuffle` | Randomise order each epoch | `True` for train, `False` for val |
| `num_workers` | CPU threads for loading | 2–4 on Colab |
| `pin_memory` | Faster CPU→GPU memory transfer | `True` when using GPU |
| `drop_last` | Drop final batch if incomplete | `True` when batch size consistency matters |

### Why batch size matters

- **Too small** (e.g. 4) — noisy gradient estimates, slow training, but sometimes better generalisation
- **Too large** (e.g. 2048) — stable gradients, fast training, but can overfit and requires more memory
- **Sweet spot** — 32–256 for most tasks; LLMs often use much larger effective batch sizes via gradient accumulation

---

## 4. Train vs Validation Split

Always maintain two separate dataloaders — one for training, one for validation:

```python
from torch.utils.data import random_split

train_set, val_set = random_split(dataset, [800, 200])   # 80/20 split

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)
```

**Training loader** shuffles — you want random order each epoch to prevent the model memorising sequence.

**Validation loader** does not shuffle — order doesn't matter, and consistency makes debugging easier.

### Why validation matters

The validation set is data the model **never trains on**. It tells you how well the model generalises to unseen data. Without it, you have no way of knowing whether your model is learning or just memorising.

| Training loss ↓, Val loss ↓ | Healthy learning |
|---|---|
| Training loss ↓, Val loss ↑ | Overfitting — model memorising training data |
| Both losses plateau high | Underfitting — model too simple or learning rate too low |
| Loss spikes suddenly | Learning rate too high — training unstable |

---

## 5. Loss Functions

The loss function measures how wrong the model is. Choice depends entirely on the task:

| Task | Loss function | PyTorch | Notes |
|---|---|---|---|
| Regression | Mean Squared Error | `nn.MSELoss()` | Penalises large errors heavily |
| Binary classification | Binary Cross Entropy | `nn.BCEWithLogitsLoss()` | Sigmoid built in — use this, not `BCELoss` |
| Multi-class classification | Cross Entropy | `nn.CrossEntropyLoss()` | Softmax built in |
| **Next token prediction** | **Cross Entropy** | **`nn.CrossEntropyLoss()`** | The LLM training objective |

### Why `BCEWithLogitsLoss` over `BCELoss`

`BCEWithLogitsLoss` applies the sigmoid activation internally in a numerically stable way. Using `BCELoss` with a sigmoid output can cause gradient instability for very confident predictions (values near 0 or 1). Always prefer `BCEWithLogitsLoss` for binary classification — your model outputs raw logits, and the loss handles the rest.

### Cross entropy for LLMs

At each token position, the LLM outputs a probability distribution over the entire vocabulary (~50,000 tokens). Cross entropy measures how far that distribution is from the correct token — it's high when the model assigns low probability to the right answer, low when it's confident and correct. Minimising this loss over billions of tokens is the entirety of LLM pre-training.

---

## 6. Optimisers

After `.backward()` computes gradients, the optimiser uses them to update weights. Different optimisers use different strategies.

### SGD — Stochastic Gradient Descent

The simplest update rule:

```
new_weight = old_weight - learning_rate × gradient
```

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

Pure SGD is noisy and slow — it takes the same step size regardless of gradient history. **Momentum** improves it by accumulating velocity in consistent directions:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Think of momentum like a ball rolling downhill — it builds up speed in consistent directions and dampens oscillation across ravines.

### Adam — Adaptive Moment Estimation

The dominant optimiser in deep learning. Improves on SGD in two ways:

**Adaptive learning rates** — each parameter gets its own learning rate, adjusted based on gradient history. Parameters with consistently large gradients get smaller steps; parameters with small gradients get larger steps.

**Momentum on both gradient direction and magnitude** — tracks the 1st moment (mean of gradients) and 2nd moment (mean of squared gradients), giving smoother, more reliable updates.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### AdamW — The LLM Standard

Identical to Adam but applies **weight decay correctly**. Adam's original weight decay was mathematically flawed — it applied decay after the adaptive scaling, which diluted the regularisation effect. AdamW fixes this by applying weight decay directly to the weights before the gradient update:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
```

> **Rule of thumb:** Use `AdamW` with `lr=3e-4` as your default starting point for almost every model. Switch to SGD with momentum only if you have a specific reason (some vision tasks converge better with it).

### Optimiser comparison

| Optimiser | Pros | Cons | Use when |
|---|---|---|---|
| SGD | Simple, memory efficient | Slow, sensitive to LR | Vision models, when you need to control regularisation precisely |
| SGD + momentum | Faster convergence than SGD | Still needs careful LR tuning | Classic vision training |
| Adam | Fast convergence, robust | Higher memory, can overfit | General default |
| AdamW | Fast + correct regularisation | Higher memory | LLMs, Transformers, most modern architectures |

---

## 7. Learning Rate & Scheduling

The learning rate is the single most impactful hyperparameter. Too high and training diverges; too low and it never converges.

### Finding a starting point

| Model type | Typical LR |
|---|---|
| Small MLP | 1e-3 to 1e-2 |
| CNN | 1e-3 to 1e-4 |
| Transformer / LLM | 3e-4 to 1e-4 |

### Learning rate scheduling

A fixed learning rate is rarely optimal. In practice you **schedule** it — start higher to move fast, reduce later to fine-tune:

```python
# Reduce by factor of 0.1 if val loss doesn't improve for 3 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=3
)
scheduler.step(avg_val_loss)   # call after each epoch
```

```python
# Cosine annealing — smoothly decays LR following a cosine curve
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)
scheduler.step()   # call after each epoch
```

### LLM standard schedule — warmup + cosine decay

LLMs use a two-phase schedule:

1. **Linear warmup** — ramp from near-zero to peak LR over the first ~1000 steps. Prevents large, destabilising updates at the start when weights are random.
2. **Cosine decay** — gradually reduce LR back toward zero over the rest of training. Allows fine-grained convergence at the end.

```python
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

---

## 8. The Complete Training Loop

This is the full pattern — memorise this structure, you'll use it for every model you train:

```python
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn   = nn.CrossEntropyLoss()

train_losses, val_losses = [], []

for epoch in range(num_epochs):

    # ── Training phase ──────────────────────────────────────
    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()                                   # 1. clear gradients
        pred = model(X_batch)                                   # 2. forward pass
        loss = loss_fn(pred, y_batch)                           # 3. compute loss
        loss.backward()                                         # 4. backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 5. clip gradients
        optimizer.step()                                        # 6. update weights

        train_loss += loss.item()

    # ── Validation phase ─────────────────────────────────────
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch  = X_batch.to(device)
            y_batch  = y_batch.to(device)
            pred     = model(X_batch)
            val_loss += loss_fn(pred, y_batch).item()

    # ── Logging ──────────────────────────────────────────────
    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)
    train_losses.append(avg_train)
    val_losses.append(avg_val)

    print(f"Epoch {epoch+1:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
```

### Why this exact order matters

| Step | Why |
|---|---|
| `model.train()` before training loop | Enables dropout, correct batchnorm behaviour |
| `zero_grad()` first | Old gradients must be cleared before computing new ones |
| Forward before loss | Need predictions before measuring error |
| Loss before backward | `.backward()` needs the scalar loss to differentiate |
| Clip before step | Caps exploding gradients before they corrupt weights |
| `model.eval()` + `no_grad()` for val | Disables dropout, skips graph construction — faster and correct |

### Gradient clipping

In deep networks, gradients can occasionally explode to huge values and destabilise training. Clipping caps the gradient norm at a maximum value:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Place this between `.backward()` and `.step()`. You'll see it in every serious LLM training script — it's not optional for deep Transformers.

---

## 9. Reading Your Training Curves

### What healthy training looks like

- Both train and val loss decrease together
- Val loss slightly higher than train loss — normal, since training loss is computed mid-epoch while weights are still changing
- Smooth descent, possibly with a sharp initial drop followed by a gradual tail

### Val loss lower than train loss

Not a bug — can happen when the val set is an easier random split, or when dropout is active during training but disabled during validation. If both curves are still descending in the same direction, everything is fine.

### Loss still descending at end of training

The model hasn't fully converged — increase epochs or learning rate. Loss being still in descent at the final epoch means you left performance on the table.

### Accuracy plateauing while loss keeps dropping

Normal and expected. **Loss is more sensitive than accuracy** — the model can be getting more confident about its correct predictions (lower loss) without changing which predictions it makes (flat accuracy).

---

## 10. How This Connects to LLMs

| Concept | Where it appears in LLMs |
|---|---|
| `Dataset.__getitem__` | Returns tokenised text sequences of length `seq_len` |
| `DataLoader` | Serving batches of shape `(batch_size, seq_len)` to the Transformer |
| `shuffle=True` | Randomising document order each epoch |
| `num_workers` | Parallel tokenisation — critical for large datasets |
| `CrossEntropyLoss` | Next token prediction — the core LLM training signal |
| `AdamW` | Universal LLM optimiser — GPT, BERT, LLaMA all use it |
| `lr=3e-4` | Standard starting LR for Transformer training |
| Warmup + cosine decay | Standard LLM learning rate schedule |
| Gradient clipping | Essential for Transformer stability — always `max_norm=1.0` |
| `model.train()` / `model.eval()` | Toggling dropout between training and generation |
| `torch.no_grad()` | Every inference call, every val loop, all text generation |
| Gradient accumulation | Running N batches without `zero_grad()` to simulate larger batch sizes on limited GPU memory — standard in LLM training |

### The LLM training loop in one line

```
for each batch of token sequences:
    predict next token at every position →
    compute cross entropy loss →
    backprop through all N Transformer layers →
    clip gradients →
    AdamW step →
    repeat for billions of tokens
```

Everything in this notebook is exactly that loop — just with a simple classifier instead of a Transformer.

---

*Previous concept → [03 — nn.Module & Custom Layers](./03_nn_module_custom_layers.md)*  
*Next concept → [05 — MLP Classifier from Scratch](./05_mlp_classifier.md)*
