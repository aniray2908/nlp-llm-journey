"""
inference.py — Educational Rewriter GPT
Phase 5 Capstone | Anisha Ray

Standalone inference module for the Teaching Quality Analyzer capstone.
Loads the fine-tuned LLaMA 3.2 3B LoRA model and provides rewrite
generation with audience suitability scoring.

Usage:
    from inference import load_model, rewrite

    model, tokenizer = load_model()
    result = rewrite(model, tokenizer, text, mode="simpler")
    print(result["rewrite"])
    print(result["audience_score"])
    print(result["recommendation"])
"""

import json
import re
import torch
import numpy as np
from pathlib import Path


# ============================================================
# JARGON SCORES (loaded once at module import)
# ============================================================

_JARGON_SCORES = None

def _load_jargon_scores():
    global _JARGON_SCORES
    if _JARGON_SCORES is None:
        jargon_path = Path(__file__).parent / "data/jargon/jargon_scores.json"
        with open(jargon_path, "r") as f:
            _JARGON_SCORES = json.load(f)
    return _JARGON_SCORES


# ============================================================
# MODE-SPECIFIC SYSTEM PROMPTS
# ============================================================

MODE_PROMPTS = {
    "default": """You are an expert educational content rewriter.
Rewrite the text to be clearer and easier to understand.
Improve readability while preserving the original meaning exactly.
Output ONLY the rewrite, nothing else.""",

    "simpler": """You are an expert educational content rewriter.
Rewrite the text so a 16-year-old with no technical background can understand it.
Replace ALL technical terms with everyday words.
Use short sentences. Maximum 15 words per sentence.
Never use acronyms or jargon.
If a technical term is unavoidable, explain it in brackets.
Output ONLY the rewrite, nothing else.""",

    "add_example": """You are an expert educational content rewriter.
Rewrite the text and add ONE concrete example that directly illustrates the concept.
The example MUST be about the exact same topic as the input text.
Do not use unrelated examples from different domains.
Format: original explanation + "For example, ..."
Output ONLY the rewrite with example, nothing else.""",

    "concise": """You are an expert educational content rewriter.
Rewrite the text using fewer words while keeping the same meaning.
Remove redundancy and filler phrases.
Do not remove important information.
Output ONLY the shorter rewrite, nothing else.""",

    "step_by_step": """You are an expert educational content rewriter.
Break down HOW THIS CONCEPT WORKS into clear numbered steps.
Each step must explain part of the mechanism described in the original text.
Do NOT explain history. Do NOT add information not in the original text.
Format strictly as:
1. [first part of the mechanism]
2. [second part]
3. [third part]
Output ONLY the numbered steps, nothing else.""",

    "add_analogy": """You are an expert educational content rewriter.
Rewrite the text and add a real-world analogy that makes the concept concrete.
The analogy must map clearly to the concept in the original text.
Explain why the analogy applies.
Output ONLY the rewrite with analogy, nothing else.""",
}


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    base_model_name="meta-llama/Llama-3.2-3B",
    adapter_name="ray-2908/educational-rewriter-lora",
    load_in_4bit=True,
):
    """
    Load the fine-tuned LLaMA 3.2 3B model with LoRA adapters.

    Args:
        base_model_name: HuggingFace base model identifier
        adapter_name: HuggingFace LoRA adapter identifier
        load_in_4bit: Whether to use 4-bit quantisation (QLoRA)

    Returns:
        model, tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"Loading base model in 4-bit...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        print(f"Loading base model in full precision...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
        )

    print(f"Loading LoRA adapters from {adapter_name}...")
    model = PeftModel.from_pretrained(base_model, adapter_name)
    model.eval()

    print(f"Model loaded successfully!")
    return model, tokenizer


# ============================================================
# AUDIENCE SUITABILITY SCORE
# ============================================================

def _flesch_kincaid_grade(text):
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = text.split()
    if not sentences or not words:
        return 0
    def count_syllables(word):
        word = word.lower().strip(".,!?;:")
        count = len(re.findall(r'[aeiou]+', word))
        if word.endswith('e') and len(word) > 2:
            count -= 1
        return max(1, count)
    total_syllables = sum(count_syllables(w) for w in words)
    avg_words_per_sentence = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)
    grade = (0.39 * avg_words_per_sentence) + (11.8 * avg_syllables_per_word) - 15.59
    return max(0, grade)

def _compute_jargon_density(text, jargon_scores):
    text_clean = re.sub(r'[^a-z\\s]', ' ', text.lower())
    words = [w for w in text_clean.split() if len(w) > 2]
    if not words:
        return 0.0
    scores = [jargon_scores.get(word, 1.0) for word in words]
    return sum(scores) / len(scores)

def _compute_sentence_complexity(text):
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.5
    avg_sent_len = np.mean([len(s.split()) for s in sentences])
    sent_len_score = min(1.0, avg_sent_len / 40)
    subordinate_markers = ['which', 'that', 'although', 'because', 'while',
                           'whereas', 'however', 'therefore', 'furthermore']
    text_lower = text.lower()
    marker_count = sum(text_lower.count(m) for m in subordinate_markers)
    clause_density = min(1.0, marker_count / max(len(sentences), 1) / 3)
    return (sent_len_score * 0.6) + (clause_density * 0.4)

def _compute_concept_density(text):
    words = text.split()
    if not words:
        return 0.5
    technical_indicators = sum(1 for w in words if
                               (len(w) > 8 and w[0].isupper()) or
                               (len(w) > 10))
    return min(1.0, technical_indicators / max(len(words), 1) * 5)

def compute_audience_score(text):
    """
    Compute audience suitability score (1-10).
    Higher = more accessible.

    Returns:
        float: Score from 1 (expert) to 10 (child-friendly)
    """
    jargon_scores = _load_jargon_scores()

    fk_grade = _flesch_kincaid_grade(text)
    readability_score = max(1, min(10, 10 - (fk_grade * 0.56)))

    jargon_density = _compute_jargon_density(text, jargon_scores)
    jargon_score = max(1, min(10, 10 - (jargon_density * 9)))

    complexity = _compute_sentence_complexity(text)
    complexity_score = max(1, min(10, 10 - (complexity * 9)))

    concept_density = _compute_concept_density(text)
    concept_score = max(1, min(10, 10 - (concept_density * 9)))

    final_score = (
        0.40 * readability_score +
        0.25 * jargon_score +
        0.20 * complexity_score +
        0.15 * concept_score
    )

    return round(final_score, 1)

def get_audience_label(score):
    """Convert score to human-readable audience label."""
    if score >= 9:
        return "All ages", "Suitable for middle school and above"
    elif score >= 7:
        return "High school level", "Suitable for ages 14+"
    elif score >= 5:
        return "Undergraduate level", "Requires some background knowledge"
    elif score >= 3:
        return "Graduate level", "Requires bachelor's-level knowledge"
    else:
        return "Expert level", "Requires deep domain expertise"

def get_recommendation(score):
    """Generate simplification recommendation based on score."""
    if score >= 8:
        return None
    elif score >= 6:
        return "This content suits undergraduate-level readers. Want a version for high school students?"
    elif score >= 4:
        return "This content requires graduate-level knowledge. Want a version for undergraduates?"
    else:
        return "This content is highly technical. Want a version for a general audience?"


# ============================================================
# REWRITE GENERATION
# ============================================================

def rewrite(
    model,
    tokenizer,
    text,
    mode="default",
    detailed=False,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.3,
):
    """
    Generate a rewrite and compute audience suitability score.

    Args:
        model: Loaded PeftModel
        tokenizer: Loaded tokenizer
        text: Input text to rewrite
        mode: One of default, simpler, add_example, concise,
              step_by_step, add_analogy
        detailed: Whether to use detailed system prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        top_k: Top-k sampling
        repetition_penalty: Penalty for repeated tokens

    Returns:
        dict: {
            "input": original text,
            "mode": rewrite mode,
            "rewrite": generated rewrite,
            "input_score": audience score of input,
            "output_score": audience score of rewrite,
            "score_delta": improvement in accessibility,
            "audience_label": human-readable audience label,
            "recommendation": simplification recommendation or None,
        }
    """
    mode_key = mode.lower().replace(" ", "_")
    system = MODE_PROMPTS.get(mode_key, MODE_PROMPTS["default"])
    prompt = f"<|system|>\n{system}\n<|user|>\nText: {text}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    rewrite_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    input_score = compute_audience_score(text)
    output_score = compute_audience_score(rewrite_text)
    audience_label, description = get_audience_label(output_score)
    recommendation = get_recommendation(output_score)

    return {
        "input": text,
        "mode": mode,
        "rewrite": rewrite_text,
        "input_score": input_score,
        "output_score": output_score,
        "score_delta": round(output_score - input_score, 2),
        "audience_label": audience_label,
        "audience_description": description,
        "recommendation": recommendation,
    }


# ============================================================
# ITERATIVE SIMPLIFICATION LOOP
# ============================================================

def simplify_until_accessible(
    model,
    tokenizer,
    text,
    target_score=7.0,
    max_iterations=3,
):
    """
    Iteratively apply Simpler mode until target audience score is reached.

    Args:
        model: Loaded PeftModel
        tokenizer: Loaded tokenizer
        text: Input text
        target_score: Stop when score reaches this value (default 7.0 = high school)
        max_iterations: Maximum simplification iterations

    Returns:
        list of dicts: Each iteration's result
    """
    history = []
    current_text = text

    for i in range(max_iterations + 1):
        mode = "default" if i == 0 else "simpler"
        result = rewrite(model, tokenizer, current_text, mode=mode)
        result["iteration"] = i
        history.append(result)

        if result["output_score"] >= target_score:
            break

        current_text = result["rewrite"]

    return history
