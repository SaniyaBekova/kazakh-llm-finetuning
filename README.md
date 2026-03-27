# Kazakh Machine Storytelling – LLM Fine-Tuning

**Group project** | UMass Amherst, COMPSCI 685 (Advanced NLP), Fall 2025  
**Authors:** Ananya Srivastava · Pablo Ghezzi · Saniya Bekova

---

## Overview

This project explores **controllable generation of Kazakh fairy tales** using a parameter-efficient fine-tuning approach on an open-weight Kazakh language model. We address a real gap in low-resource NLP: Kazakh is significantly underrepresented in LLM training data, and existing multilingual models struggle with narrative coherence and cultural grounding in Kazakh.

We compare four systems — prompted baseline, instruction-tuned generator, hierarchical Planner→Realizer pipeline, and DPO post-training — and evaluate them with automatic metrics, human evaluation, and LLM-as-judge (GPT-4, Gemini).

---

## Repository Structure

```
kazakh-llm-finetuning/
├── data/
│   └── StoryScraper.ipynb          # Web scraping + MinHash deduplication + dataset pipeline
├── training/
│   ├── Instruction_tuning.ipynb    # Model 1: SFT (QLoRA fine-tuning)
│   └── DPO.ipynb                   # Model 3: DPO post-training (preference optimization)
├── evaluation/
│   └── Evaluation_part1.ipynb      # BERTScore, perplexity/loss, LLM-as-judge evaluation
└── README.md
```

---

## Base Model

[`issai/LLama-3.1-KazLLM-1.0-8B`](https://huggingface.co/issai/LLama-3.1-KazLLM-1.0-8B) — developed by the Institute of Smart Systems and Artificial Intelligence (ISSAI), Nazarbayev University. Trained on Kazakh, Russian, and English.

---

## Dataset

- **Scraped:** 1,000+ Kazakh fairy tales from 3 sources:
  - [ertegiler.kz](https://ertegiler.kz) (156 used)
  - [balalaralemi.kz](https://balalaralemi.kz) (44 used)
  - [wikisource.org](https://wikisource.org) (40 used)
- **After deduplication & cleaning:** 240 stories used for training (GPU memory constraints)
- **Deduplication method:** MinHash Jaccard similarity (threshold 0.9, 256 permutations)
- **Control tags per story:** `<AGE=...> <THEME=...> <LENGTH=...> <SEED=...>`
- **Train/Val/Test split:** 80/10/10 (192 / 24 / 24)

### Structured Plans
- **40 stories manually annotated** with gold plans: setting, characters, plot phases (beginning → conflict → development → climax → resolution), moral
- Remaining stories: silver plans generated via KazLLM-8B in-context prompting, validated programmatically

---

## Models

| Model | Description |
|-------|-------------|
| **Model 0** | Prompted baseline (no fine-tuning) |
| **Model 1** | Instruction-tuned: CTRL tags → story (SFT, QLoRA) |
| **Model 2** | Hierarchical Planner→Realizer: CTRL → plan → story |
| **Model 3** | DPO post-training on Model 2 Realizer (preference optimization) |

All trained models use:
- **LoRA adapters** (only adapter weights updated)
- **4-bit quantization** (QLoRA via bitsandbytes)
- **AdamW-style optimization** (paged variants)
- **A100 GPU** on Google Colab

---

## Training Pipeline

### SFT (Model 1) — `Instruction_tuning.ipynb`
- Fine-tunes KazLLM on CTRL tags → story pairs
- Causal LM objective with completion-only loss (prompt tokens masked)
- Max sequence length: 2048 tokens

### DPO (Model 3) — `DPO.ipynb`
- Initialized from Model 2 (Realizer) checkpoint
- Preference pairs constructed by sampling K=3 candidates per prompt, scored with **BERTScore F1** (xlm-roberta-base, lang="kk")
- Chosen = highest BERTScore candidate; Rejected = lowest
- Lightweight post-training: small learning rate, frozen reference model

---

## Evaluation — `Evaluation_part1.ipynb`

### Automatic Metrics
| Metric | Description |
|--------|-------------|
| Grammaticality | Ridge classifier on Kazakh sentence embeddings |
| Narrative Productivity | Flesch / Flesch-Kincaid / SMOG index (normalized) |
| Contextuality | Mean cosine similarity between sequential sentence pairs |
| Temporal Ordering | Relatedness decay across sentence distances |
| Connection to Prompt | Sentence embedding similarity to control prompt |
| ROUGE-1 / ROUGE-L | Overlap with reference stories |
| Correctness | Kazakh spell-check (valid word ratio) |
| Sentiment | Sentiment consistency relative to seed sentence |

### Human Evaluation
Native Kazakh speakers rated stories on: prompt relevance, clarity, Kazakh folklore style, and creativity (1–5 scale).

### LLM-as-Judge
GPT-4 and Gemini rated: relevance, clarity, fairy-tale style, creativity, orthographic errors, morphological errors.

### Key Result
The **Hierarchical Planner→Realizer** model performed best overall: highest contextuality (0.642), correctness (0.838), ROUGE scores (0.299), and nearly perfect scores from Gemini (5.0 relevance, 4.8 style).

---

## How to Reproduce

> **Hardware requirement:** A100 GPU (40GB) recommended. All notebooks were developed and run on **Google Colab Pro**. Training on smaller GPUs will require reducing `max_seq_length` and batch sizes.

### Step 1 — Data Collection
Run `data/StoryScraper.ipynb` to scrape and deduplicate Kazakh fairy tales from the three sources. Output: `train_with_tags.jsonl` with control tags per story.

> If you skip scraping, contact the authors for the preprocessed dataset.

### Step 2 — SFT Training (Model 1)
Run `training/Instruction_tuning.ipynb`:
- Mounts Google Drive, loads `dataset_v4_final.json`
- Splits data 80/10/10, saves `instruction_train.jsonl` / `val.jsonl`
- Fine-tunes KazLLM-8B with QLoRA (4-bit, LoRA r=32)
- Saves adapter weights to `/models/instruction_tuned_kazllm/`

### Step 3 — DPO Training (Model 3)
Run `training/DPO.ipynb`:
- Loads Model 2 (Planner→Realizer) checkpoint from Drive
- Samples K=3 candidates per prompt, scores with BERTScore F1
- Runs DPO training, saves adapter to `/models/realizer_dpo_kazllm_v3/`

### Step 4 — Evaluation
Run `evaluation/Evaluation_part1.ipynb`:
- Loads all model checkpoints
- Computes automatic metrics (grammaticality, contextuality, ROUGE, BERTScore)
- Runs LLM-as-judge evaluation (GPT-4, Gemini)
- Outputs results to `evaluation_part1.csv`

### Expected Runtime (A100)
| Step | Approximate Time |
|------|-----------------|
| Scraping + deduplication | ~30 min |
| SFT training (3 epochs) | ~2–3 hours |
| DPO training | ~1 hour |
| Evaluation | ~1–2 hours |

---



```
transformers
peft
bitsandbytes
trl
datasets
bert-score
accelerate
pandas
```

Install via:
```bash
pip install transformers peft bitsandbytes trl datasets bert-score accelerate pandas
```

> **Note:** Training requires an A100 GPU (or equivalent). All notebooks were developed and run on Google Colab Pro.

---

## Contributions

| Member | Contributions |
|--------|--------------|
| **Ananya Srivastava** | Dataset labelling, silver set generation, final dataset pipeline, Model 2 (Planner→Realizer), report writing |
| **Pablo Ghezzi** | Dataset generation, evaluation pipeline, report writing |
| **Saniya Bekova** | Source verification, dataset generation, **manual annotation of 40 stories** (gold plans), **Model 1 (SFT)** and **Model 3 (DPO)** implementation, manual and LLM-as-judge evaluation, perplexity/loss and BERTScore calculation |

---

## Citation / Course

This project was completed for **COMPSCI 685: Advanced NLP** at the University of Massachusetts Amherst (Fall 2025).

Base model: [ISSAI KazLLM](https://issai.nu.edu.kz/kazllm/)  
QLoRA: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)  
DPO: [Rafailov et al., 2024](https://arxiv.org/abs/2305.18290)  
BERTScore: [Zhang et al., 2020](https://arxiv.org/abs/1904.09675)
