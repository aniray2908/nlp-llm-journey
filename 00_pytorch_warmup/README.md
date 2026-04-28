# Phase 0 — PyTorch Warm-up

> Building solid PyTorch fundamentals before touching NLP. Everything downstream — Transformers, LLMs, fine-tuning — depends on fluent tensor operations and a clean training loop.

---

## What's in This Phase

This phase builds up the toolkit needed to train any neural network in PyTorch — from raw tensor operations all the way to a working classifier on real image data.

| # | Topic | Type |
|---|---|---|
| 01 | Tensors, slicing, broadcasting, GPU basics | Learn |
| 02 | Autograd — how gradients are tracked and computed | Learn |
| 03 | Linear regression from scratch | Build |
| 04 | nn.Module and custom layers | Learn |
| 05 | DataLoader, training loop, optimisers | Learn |
| 06 | MLP classifier on MNIST | Build |

---

## Folder Structure

```
00_pytorch_warmup/
├── concepts/
│   ├── 01_tensors_slicing_broadcasting_gpu.md
│   ├── 02_autograd.md
│   ├── 03_nn_module_custom_layers.md
│   └── 04_dataloader_training_loop_optimisers.md
└── demos/
    ├── 01_tensors_slicing_broadcasting_gpu.ipynb
    ├── 02_autograd.ipynb
    ├── 03_linear_regression_from_scratch.ipynb
    ├── 04_nn_module_custom_layers.ipynb
    ├── 05_dataloader_training_loop_optimisers.ipynb
    └── 06_mlp_mnist_classifier.ipynb
```

The `concepts/` folder contains detailed written notes for each topic. The `demos/` folder contains the hands-on Colab notebooks where each concept was practised.

---

## Key Concepts Covered

**Tensors** — the core data structure of PyTorch. Shape, dtype, device, slicing, broadcasting rules, and reshape patterns. Built intuition for the `(batch, seq_len, embed_dim)` shape that flows through every Transformer.

**Autograd** — PyTorch's automatic differentiation engine. Computation graphs, the chain rule, `requires_grad`, `torch.no_grad()`, `.detach()`, and the gradient accumulation gotcha (and feature).

**nn.Module** — the base class for every neural network. Custom layers using `nn.Parameter`, stacking with `nn.Sequential` and `nn.ModuleList`, switching between train/eval modes, saving and loading state dicts.

**Training loops** — `Dataset`, `DataLoader`, batching, shuffling. The five-step training pattern (`zero_grad → forward → loss → backward → step`). Optimisers from SGD to AdamW. Loss functions for regression and classification. Gradient clipping for stability.

---

## Projects Built

**Linear regression from scratch** — recovered `y = 2x + 1` from noisy data using both manual gradient descent and `nn.Linear` + SGD, verifying that the abstractions match the underlying math.

**Two-layer MLP on `sin(x)`** — demonstrated that non-linearity is essential by fitting a curve a single linear layer could never learn.

**Binary classifier on gaussian clusters** — built the first proper end-to-end pipeline with `Dataset`, `DataLoader`, train/val split, AdamW optimiser, and a decision boundary visualisation.

**MLP classifier on MNIST** — Phase 0 capstone. ~98% test accuracy on handwritten digit recognition, with confusion matrix analysis showing the model fails exactly on genuinely ambiguous digits (4↔9, 7↔1, 3↔8).

---

## What This Phase Set Up

By the end of Phase 0, the following workflow felt natural:

```python
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn   = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            ...
```

This pattern is the heartbeat of every neural network training run — including LLMs. The architecture changes, the data changes, the scale changes. The loop stays the same.

---

## What's Next

Phase 1 — NLP Fundamentals → Tokenisation, embeddings, language modelling, and the classical NLP pipeline that everything LLM-related is built on.
