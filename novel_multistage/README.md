
---

#  **PHASE 2: NOVEL MULTI-STAGE FACTUALITY-CONSTRAINED TRAINING **


This phase introduces a **novel 3-stage progressive training framework** that explicitly integrates **factuality constraints during training**, rather than relying on post-hoc correction.

---

## Motivation

Phase 1 revealed that:
- Entity preservation plateaus at ~70%
- Temporal consistency remains ~77%
- ROUGE improvements do not guarantee factual correctness

**Goal:** Teach the model factuality *during training*.

---

## Model & Dataset

- **Base Model:** LED-Large-16384-ArXiv
- **Dataset:** NewsSumm (Indian English)
- **Training Samples:** 2,000
- **Validation Samples:** 1,000
- **Platform:** Google Colab (Tesla T4, 16GB)
- **Training Time:** ~2.5 hours

---

## Novel 3-Stage Training Framework

### Stage 1: Warmup (Fluency & Domain Adaptation)
- Loss: 1.1087 → 0.5086
- Objective: Learn Indian English structure
- Factuality losses disabled

---

### Stage 2: Factuality Enforcement
- Entity Loss activated
- Temporal Loss activated

| Metric | Value |
|------|------|
| Entity Loss | 0.0311 |
| Entity Preservation | **96.9%** |
| Temporal Loss | 0.0007 |
| Temporal Consistency | **99.93%** |

---

### Stage 3: Refinement (Balanced Objectives)

| Component | Value |
|--------|-------|
| Total Loss | 0.2878 |
| Entity Preservation | **96.74%** |
| Temporal Consistency | **99.90%** |

---

## Dynamic Loss Weighting (Novel)

| Stage | CE | Entity | Temporal | Semantic |
|----|----|-------|----------|----------|
| 1 | 1.0 | 0.0 | 0.0 | 0.0 |
| 2 | 0.5 | 0.3 | 0.2 | 0.0 |
| 3 | 0.4 | 0.2 | 0.2 | 0.2 |

---

## Indian-Specific Loss Functions

### IndianEntityLoss
- Tracks 56 Indian entities
- Handles political, numerical, administrative terms

### TemporalConsistencyLoss
- Detects hallucinated or shifted dates

### SemanticFidelityLoss
- SBERT-based semantic alignment

---

## Results Comparison

| Metric | Phase 1 (Baseline) | Phase 2 (Proposed) |
|-----|----------------|----------------|
| Entity Preservation | ~70% | **96.7%** |
| Temporal Consistency | ~77% | **99.9%** |
| Training Style | Single-stage | Multi-stage |
| Factuality Awareness | Post-hoc | Integrated |

---

## Code Structure
novel_multistage/
 novel_losses.py
 train_multistage.py
 results/


---

## Reproducibility

```bash
pip install torch transformers sentence-transformers spacy
python -m spacy download en_core_web_sm

cd novel_multistage
python code/train_multistage.py \
  --train_data ../data/processed/train_2k.csv \
  --val_data ../data/processed/val_1k.csv

