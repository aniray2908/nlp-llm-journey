# Phase 5 — Educational Rewriter GPT

## Overview

This phase documents the development of a fine-tuned language model designed to rewrite confusing educational content into clearer, more accessible alternatives. The model serves as the core inference engine for the Teaching Quality Analyzer capstone project and is published independently as a reusable NLP component.

The project covers the complete machine learning pipeline: dataset curation, synthetic data generation, model fine-tuning, evaluation, and deployment to HuggingFace Hub.

---

## Model

| Property | Value |
|----------|-------|
| **Base model** | meta-llama/Llama-3.2-3B |
| **Fine-tuning method** | QLoRA (4-bit quantisation + LoRA) |
| **LoRA rank** | 16 |
| **Trainable parameters** | 3,407,872 (0.45% of total) |
| **Training examples** | 846 |
| **Training time** | 150 minutes (NVIDIA Tesla T4) |
| **HuggingFace** | [ray-2908/educational-rewriter-lora](https://huggingface.co/ray-2908/educational-rewriter-lora) |
| **License** | Apache 2.0 |

---

## Rewrite Modes

The model supports six targeted rewrite modes, each addressing a specific clarity need in educational content.

| Mode | Description | Use case |
|------|-------------|----------|
| **Default** | General clarity improvement | Any dense or poorly structured text |
| **Simpler** | Jargon removal, plain language | Non-expert or younger audiences |
| **Add Example** | Adds a domain-relevant concrete example | Abstract concepts that need grounding |
| **Concise** | Reduces word count, preserves meaning | Verbose or repetitive content |
| **Step by Step** | Breaks mechanism into numbered steps | Processes, algorithms, how-to content |
| **Add Analogy** | Adds a real-world comparison | Abstract or counterintuitive concepts |

---

## Repository Structure

```
05_rewriter_gpt/
├── README.md
├── data/
│   ├── raw/
│   │   ├── passages.json              — 201 raw collected passages
│   │   └── passages_clean.json        — 141 cleaned passages (66 Wikipedia + 75 arXiv)
│   ├── processed/
│   │   ├── rewrites.json              — 846 (input, mode, output) training triplets
│   │   └── dataset/                   — HuggingFace Dataset format
│   │       ├── train/                 — 612 examples (70%)
│   │       ├── validation/            — 138 examples (17%)
│   │       └── test/                  — 96 examples (13%)
│   └── jargon/
│       ├── word_frequencies.json      — Full word frequency data (10,418 words)
│       ├── jargon_scores.json         — Lightweight jargon scores for inference
│       └── frequency_distribution.png — Tier distribution visualisation
├── demos/
│   ├── 01_collect_passages.ipynb      — Wikipedia + arXiv passage collection
│   ├── 02_generate_rewrites.ipynb     — Claude API rewrite generation
│   ├── 03_finetune_llama.ipynb        — QLoRA fine-tuning on Kaggle (T4 GPU)
│   └── 04_jargon_frequency_list.ipynb — Simple English Wikipedia jargon system
└── results/
    └── training_curves.png            — Train and validation loss across 3 epochs
```

---

## Pipeline

### Step 1 — Dataset Collection

**Notebook:** `demos/01_collect_passages.ipynb`

Collected 141 confusing educational passages from two sources:

- **Wikipedia** (66 passages) — Technical article sections across computer science, biology, physics, mathematics, economics, chemistry, and medicine. Filtered for density, sentence length, and absence of markup artifacts.
- **arXiv** (75 passages) — Academic paper abstracts across machine learning, NLP, quantum computing, economics, and statistics. Naturally dense and jargon-heavy.

Passages were filtered for length (30-300 words), duplicate removal, and quality before proceeding to generation.

### Step 2 — Synthetic Data Generation

**Notebook:** `demos/02_generate_rewrites.ipynb`

Each passage was rewritten in all 6 modes using the Claude API (claude-sonnet-4-5), following the synthetic data generation methodology established in Stanford Alpaca (Taori et al., 2023).

Key design decisions:
- **Dynamic prompting:** Each example was randomly assigned either a short or detailed system prompt (49% / 51% split), exposing the model to both instruction styles during training
- **Manual seed examples:** A small set of manually written examples anchored generation quality
- **Auto-save with resume:** The generation script saved progress after every passage, enabling safe resumption on interruption

**Generation statistics:**

| Metric | Value |
|--------|-------|
| Total API calls | 846 |
| Total tokens | 401,756 (242,618 input + 159,138 output) |
| Total cost | $3.12 |
| Generation time | ~90 minutes |
| Concise mode check | 141/141 outputs shorter than input |
| Step by Step check | 77/141 outputs contained numbered steps |

### Step 3 — Fine-tuning

**Notebook:** `demos/03_finetune_llama.ipynb`  
**Platform:** Kaggle (NVIDIA Tesla T4, 16GB)

Fine-tuned LLaMA 3.2 3B using QLoRA via TRL's SFTTrainer. The base model was loaded in 4-bit NF4 quantisation, reducing memory from approximately 12GB to under 4GB. LoRA adapters were applied to all four attention projection layers.

**Training configuration:**

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 3 |
| Learning rate | 2e-4 (cosine schedule) |
| Batch size | 2 + gradient accumulation 8 |
| Effective batch size | 16 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Max sequence length | 512 |
| Training precision | BFloat16 |

**Training results:**

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1 | 1.737 | 1.183 |
| 2 | 1.149 | 1.086 |
| 3 | 1.044 | 1.065 |

Train/validation gap at epoch 3: **0.021** — indicating minimal overfitting on the 612-example training set.

![Training Curves](results/training_curves.png)

### Step 4 — Jargon Detection System

**Notebook:** `demos/04_jargon_frequency_list.ipynb`

Built a domain-agnostic jargon detection system using Simple English Wikipedia as a frequency benchmark. Simple English Wikipedia is written for non-native speakers and younger audiences, making it a principled proxy for accessibility.

**Corpus:** 59 articles, 112,433 words across science, technology, history, nature, health, and everyday topics.

**Supplemented with:**
- Dolch word list (220 common sight words)
- Fry word list (first 500 most common English words)

**Tier thresholds (calibrated for corpus size):**

| Tier | Frequency | Jargon score | Words | % |
|------|-----------|--------------|-------|---|
| Common | 50+ | 0.0 | 789 | 7.6% |
| Familiar | 10-49 | 0.3 | 993 | 9.5% |
| Technical | 2-9 | 0.7 | 4,042 | 38.8% |
| Jargon | <2 | 1.0 | 4,594 | 44.1% |

**Validation:**

| Text | Jargon density | Tier |
|------|---------------|------|
| "The dog sat in the park" | 0.222 | Familiar |
| "Photosynthesis is the process by which plants make food" | 0.067 | Common |
| "Backpropagation computes gradients via chain rule" | 0.580 | Technical |
| "The eigendecomposition of the Hessian matrix" | 0.555 | Technical |

---

## Evaluation

### Qualitative Mode Assessment

Evaluation was conducted on a held-out test passage (85 words on Transformer architecture) across all 6 modes using the fine-tuned inference function.

| Mode | Result | Notes |
|------|--------|-------|
| Default | Good | Clear structure, accessible language, appropriate length |
| Simpler | Partial | Reduces complexity but occasionally retains technical vocabulary |
| Add Example | Partial | Adds examples but domain relevance is inconsistent |
| Concise | Good | Consistently produces shorter outputs (49% reduction observed) |
| Step by Step | Good | Produces structured numbered steps following source mechanism |
| Add Analogy | Good | Generates relevant real-world comparisons with clear mappings |

### Known Limitations

**Simpler mode** occasionally retains jargon from the source text. Root cause: training data generated by Claude which itself uses technical vocabulary in rewrites. Proposed fix: regenerate Simpler mode training pairs with stricter jargon elimination constraints.

**Add Example mode** sometimes adds examples from unrelated domains. Root cause: insufficient domain-specificity requirements in generation prompts. Proposed fix: regenerate Add Example pairs with explicit domain-matching constraints.

**Model size:** At 3B parameters, the model shows strong instruction-following on most modes but may struggle with highly complex or ambiguous inputs. Upgrading to a 7B base model is expected to improve consistency.

**Factual accuracy:** The model may introduce inaccuracies in rewrites. All outputs should be reviewed before publication in high-stakes contexts.

---

## Audience Suitability Score

The jargon detection system feeds into a formula-based audience suitability score (1-10) that indicates the accessibility level of any rewritten text.

```
Score = readability (40%) + jargon density (25%) + sentence complexity (20%) + concept density (15%)

Scale:
  1  — Expert / researcher level
  3  — Graduate level
  5  — Undergraduate level
  7  — High school level
  10 — Accessible to a curious 10-year-old
```

This score powers the pop-up recommendation and iterative simplification loop in the Teaching Quality Analyzer capstone.

---

## Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config,
    device_map="auto",
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "ray-2908/educational-rewriter-lora")
model.eval()
```

See `demos/03_finetune_llama.ipynb` for the complete inference function with all 6 mode-specific system prompts.

---

## Future Work

1. **Regenerate Simpler and Add Example training pairs** with stricter constraints and retrain (~$0.50 + 2.5 hours)
2. **Upgrade base model to LLaMA 3.2 7B** for improved instruction following across all modes
3. **Expand training dataset** to 2,000+ examples for better generalisation across domains
4. **Implement audience suitability scoring** — integrate jargon density with readability metrics into the full 1-10 formula
5. **Mode-specific LoRA adapters** — separate adapter per mode for higher per-mode quality

---

## Dependencies

```bash
pip install transformers peft accelerate bitsandbytes trl datasets sentencepiece huggingface_hub anthropic
```

---

## Related Resources

- **HuggingFace model:** [ray-2908/educational-rewriter-lora](https://huggingface.co/ray-2908/educational-rewriter-lora)
- **Capstone project:** [teaching-quality-analyzer](https://github.com/ray-2908/teaching-quality-analyzer) *(coming soon)*
- **Base model:** [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
- **LoRA paper:** [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- **QLoRA paper:** [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)
- **Alpaca methodology:** [Taori et al., 2023](https://crfm.stanford.edu/2023/03/13/alpaca.html)

---

## Citation

```bibtex
@misc{ray2026rewriter,
  author    = {Anisha Ray},
  title     = {Educational Content Rewriter — LLaMA 3.2 3B LoRA},
  year      = {2026},
  url       = {https://huggingface.co/ray-2908/educational-rewriter-lora}
}
```

---

*Phase 5 — Educational Rewriter GPT | Part of the NLP/LLM Learning Journey*  
*Previous: [Phase 4 — LLM Internals and Fine-tuning](../04_llm_internals_finetuning/)*  
*Next: Teaching Quality Analyzer Capstone*
