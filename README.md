# Multi-Stage Factuality-Constrained Training for Indian English News Summarization

**Novel Contribution:** Progressive multi-stage training with Indian English-specific factuality constraints.

##  Research Overview

This repository presents a **two-phase research approach**:

### **Phase 1: Baseline Investigation** (Motivation)
- Fine-tuned LED on NewsSumm (253K articles)
- Evaluated 10 state-of-the-art models
- **Found:** Standard models achieve only 70% entity preservation, 77% temporal consistency
- **Gap Identified:** Factuality not integrated in training

 See [`baseline_investigation/`](baseline_investigation/)

### **Phase 2: Novel Multi-Stage Framework** (Solution)
- Developed 3-stage training with dynamic loss weighting
- Integrated factuality constraints **during training**
- **Results:** 96.7% entity preservation, 99.9% temporal consistency
- **Improvement:** +26.7% entity preservation, +22.9% temporal consistency

 See [`novel_multistage/`](novel_multistage/)

---

##  Key Results

### Baseline (Phase 1) vs Novel Approach (Phase 2)

| Metric | Baseline (LED) | Novel Multi-Stage | Improvement |
|--------|---------------|-------------------|-------------|
| **ROUGE-L** | 29.21% | 41.2% | **+12%** |
| **Entity Preservation** | 70% | 96.7% | **+26.7%** |
| **Temporal Consistency** | 77% | 99.9% | **+22.9%** |

### Multi-Stage Training Progress
```
Stage 1 (Warmup):     Loss 1.108 → 0.509  (Fluency)
Stage 2 (Factuality): Loss 0.509 → 0.288  (Entity + Temporal)
Stage 3 (Refinement): Loss 0.288 → 0.288  (Balanced)
```

---

##  What's Novel

1. **Multi-Stage Training Architecture**
   - Stage 1: Warmup (fluency)
   - Stage 2: Factuality focus
   - Stage 3: Multi-objective refinement

2. **Dynamic Loss Weighting**
```
   Stage 1: CE=1.0, Entity=0.0, Temporal=0.0
   Stage 2: CE=0.5, Entity=0.3, Temporal=0.2
   Stage 3: CE=0.4, Entity=0.2, Temporal=0.2, Semantic=0.2
```

3. **Indian English-Specific Losses**
   - Entity Preservation Loss (56 Indian terms)
   - Temporal Consistency Loss (date verification)
   - Semantic Fidelity Loss

---

##  Quick Start

### Installation
```bash
git clone https://github.com/Kavyahegde-2609/Multi-Stage Factuality-Constrained Training for Indian English News Summarization.git
cd Multi-Stage Factuality-Constrained Training for Indian English News Summarization
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Reproduce Baseline Results (Phase 1)
```bash
cd baseline_investigation
python baseline_evaluator.py
```

### Train Novel Multi-Stage Model (Phase 2)
```bash
cd novel_multistage
python novel_training.py \
    --train_data ../data/train_2k.csv \
    --val_data ../data/val_1k.csv
```

---

##  Repository Structure
```
baseline_investigation/      # Phase 1: Motivation
    10 baseline models evaluation
  Standard LED fine-tuning

 novel_multistage/           # Phase 2: Novel contribution
   Multi-stage training
    Custom factuality losses

 data/                       # NewsSumm preprocessing
 evaluation/                 # Evaluation scripts
 paper/                      # Research paper
 notebooks/                  # Jupyter notebooks
```

---



---

##  Acknowledgments

- **Dataset:** NewsSumm (Motghare et al., 2025)
- **Base Model:** LED-Large-16384 (Beltagy et al., 2020)

##  License

MIT License



