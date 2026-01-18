# Multi-Stage Factuality-Constrained Training for Indian English News Summarization

A novel 3-stage training framework with Indian-specific factuality constraints.

## Quick Summary

**Problem:** Generic summarization models show substantial degradation in Indian entity preservation and notable temporal inconsistencies in news summaries.

**Solution:** A 3-stage progressive training framework with factuality-aware loss functions.

**Results:**

| Metric | Baseline | Novel | Improvement |
|------|---------|-------|-------------|
| Entity Preservation | ~70% | 96.7% | +26.7% |
| Temporal Consistency | ~77% | 99.9% | +22.9% |

*Improvements are computed using the proposed factuality evaluation module on a held-out evaluation set.*

---

## Two-Phase Approach

├── baseline_investigation/ # Phase 1: 10-model evaluation
├── novel_multistage/ # Phase 2: Novel training
│ ├── code/
│ │ ├── novel_losses.py
│ │ └── train_multistage.py
│ └── results/
├── data/processed/ # NewsSumm dataset
└── README.md

---

### Phase 1: Baseline Investigation

Evaluation of 10 transformer-based summarization models on the NewsSumm dataset.

- Best baseline: **BART-Large-CNN** (30.83% ROUGE-L)
- Observed gap: entity preservation (~70%), temporal consistency (~77%)

[Details →](baseline_investigation/README.md)

---

### Phase 2: Novel Multi-Stage Training

A 3-stage progressive framework with dynamic loss weighting:

- **Stage 1:** Warmup (fluency learning)
- **Stage 2:** Factuality (entity and temporal constraints)
- **Stage 3:** Refinement (balanced multi-objective optimization)

[Details →](novel_multistage/README.md)

---

## Quick Start

```bash
git clone https://github.com/Kavyahegde-2609/Multi-Stage-Factuality-Constrained-Training-for-Indian-English-News-Summarization.git
cd Multi-Stage-Factuality-Constrained-Training-for-Indian-English-News-Summarization

pip install -r requirements.txt

# Phase 1: Baseline evaluation
cd baseline_investigation
python baseline_evaluator.py

# Phase 2: Multi-stage training
cd ../novel_multistage
python code/train_multistage.py

## Key Contributions

Multi-Stage Training Framework for Indian English news summarization

Dynamic Loss Weighting across progressive training stages

Indian-Specific Factuality Losses (entity, temporal, semantic)

##Citation
@misc{hegde2026multistage,
  title={Multi-Stage Factuality-Constrained Training for Indian English News Summarization},
  author={Kavya Mahabaleshwara Hegde},
  year={2026}
}

**Last Updated:** January 18, 2026