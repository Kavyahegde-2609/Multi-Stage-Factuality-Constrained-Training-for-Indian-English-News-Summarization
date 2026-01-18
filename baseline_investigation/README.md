# Phase 1: Baseline Investigation

This phase presents a comprehensive baseline study for **Indian English multi-document news summarization** using the **NewsSumm dataset**.  
We evaluate **10 state-of-the-art transformer models** under a **uniform evaluation setup**, followed by a **standard LED fine-tuning baseline** to establish a strong reference point for Phase 2.

---

## Dataset

- **Dataset:** NewsSumm (Indian English News)
- **Preprocessed Articles:** 253,930 (from original 317,498)
- **Test Samples:** 200
- **Evaluation Platform:** CPU / GPU (Kaggle)
- **Metrics:** ROUGE-1, ROUGE-2, ROUGE-L  
- **Factuality Analysis:** Entity Preservation, Temporal Consistency, Semantic Similarity (post-hoc)

---

## Models Evaluated ( Pretrained)

| Rank | Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-----:|-------|--------:|--------:|--------:|
| 1 | BART-Base | **46.16** | **21.67** | 29.92 |
| 2 | BART-Large-CNN | 41.02 | 21.54 | **30.83** |
| 3 | DistilBART | 42.05 | 21.31 | 30.32 |
| 4 | LED-ArXiv | 44.97|20.76|29.21|
| 5 | T5-Large | 40.26 | 18.95 | 28.45 |
| 6 | PEGASUS-CNN | 37.48 | 17.72 | 26.81 |
| 7 | T5-Base | 38.11 | 17.39 | 27.08 |
| 8 | LongT5-Base | 34.39 | 14.43 | 22.79 |
| 9 | PEGASUS-XSum | 33.27 | 13.39 | 21.82 |
|10 | mT5-Base | 14.92 | 4.38 | 10.14 |

**Best ROUGE-L:** BART-Large-CNN (30.83)

---

## Baseline LED Fine-Tuning (Standard)

To strengthen the baseline, **LED-Large-16384-ArXiv** was fine-tuned using **standard cross-entropy loss only**.

### Fine-Tuning Setup
- **Training Samples:** 10,000
- **Validation Samples:** 1,000
- **Epochs:** 2
- **Effective Batch Size:** 8
- **Training Time:** ~3.9 hours
- **Loss Reduction:**  
  - Train Loss: ~1.29 → ~0.92  
  - Validation Loss: ~1.25 → ~1.16  

This establishes a **strong neural baseline**, but **without factuality constraints**.

---

## Post-hoc Factuality Analysis (Baseline)

Using the factuality module (`factuality_module.py`), baseline outputs were analyzed.

### Observed Trends (Baseline)

- **Entity Preservation:** ~70%
- **Temporal Consistency:** ~77%
- **Semantic Similarity:** Generally high

> Observation: High semantic similarity often **masks factual errors**, such as dropped Indian entities or hallucinated dates.

### Typical Errors
- Dropped Indian terms: *crore, lakh, Lok Sabha, CM, IAS*
- Date hallucinations or shifts
- Numerically inconsistent summaries

---

## Key Takeaways from Phase 1

1. ROUGE scores alone are **insufficient** for factual reliability.
2. Pretrained models struggle with **Indian-specific entities**.
3. Temporal hallucinations persist even in high-ROUGE outputs.
4. Standard fine-tuning improves fluency but **does not solve factuality**.

These limitations directly motivate **Phase 2**.

---

## Code Structure

baseline_investigation/
├── baseline_evaluator.py
├── factuality_module.py
├── finetune_led_basic.py
├── config.py
└── results/

---

## Reproducibility

```bash
pip install -r requirements.txt
python baseline_evaluator.py --test_data data/processed/test_full.csv


Conclusion:
Phase 1 establishes a strong but factuality-limited baseline, justifying the need for a constrained training approach.