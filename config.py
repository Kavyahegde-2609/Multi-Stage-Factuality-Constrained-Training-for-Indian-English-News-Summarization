"""
Research Configuration File
==========================
Project: Factuality-Aware Domain Adaptation for Indian English News
Author: Kavya Hegde
Date: January 2025

"""

from pathlib import Path
import torch

# ==============================================================================
# PROJECT METADATA
# ==============================================================================
PROJECT_INFO = {
    "author":"Kavya Mahabaleshwara Hegde",
    "title":"Factuality-Aware Domain Adaptation for Indian English Multi-Document News Summarization",
    "github": "https://github.com/Kavyahegde-2609/factuality-aware-indian-news-summarization",
    "start_date": "January 2025",
    "dataset": "NewsSumm (253,930 articles after preprocessing from original 317,498)",
}

# ==============================================================================
# DATASET PATHS
# ==============================================================================
# Original NewsSumm: 317,498 articles (Motghare et al., 2023)
# After preprocessing: 253,930 articles
# Preprocessing: Removed duplicates, nulls, length outliers
# Splits: 90% train (228,537), 5% val (12,696), 5% test (12,697)

DATA_PATHS = {
    'test_small': Path('data/processed/test_small.csv'),
    'val_small': Path('data/processed/val_small.csv'),
    'train_small': Path('data/processed/train_small.csv'),
    'test_full': Path('data/processed/test_full.csv'),
    'val_full': Path('data/processed/val_full.csv'),
    'train_full': Path('data/processed/train_full.csv'),
}

RESULT_PATHS = {
    'baselines': Path('results/baselines'),
    'proposed': Path('results/proposed'),
    'analysis': Path('results/analysis'),
    'figures': Path('results/figures'),
}

# ==============================================================================
# STEP 1 & 2: BASELINE MODEL SELECTION
# ==============================================================================
# Methodology Step 1: Source domain models (trained on CNN/DailyMail, BBC, XSum)
# Methodology Step 2: LED and Long-T5 as primary baselines for multi-document

BASELINE_MODELS = {
    # BART models - pretrained on CNN/DailyMail (Lewis et al., 2020)
    'BART-Large-CNN': 'facebook/bart-large-cnn',
    'BART-Base': 'facebook/bart-base',
    
    # PEGASUS models - pretrained on XSum and CNN/DM (Zhang et al., 2020)
    'PEGASUS-XSum': 'google/pegasus-xsum',
    'PEGASUS-CNN': 'google/pegasus-cnn_dailymail',
    
    # T5 models - text-to-text framework (Raffel et al., 2020)
    'T5-Base': 't5-base',
    'T5-Large': 't5-large',
    
    # PRIMARY BASELINES - Long-document models (Methodology Step 2)
    # LED - Longformer Encoder-Decoder (Beltagy et al., 2020)
    # Rationale: Handles up to 16,384 tokens (suitable for multi-document)
    'LED-ArXiv': 'allenai/led-large-16384-arxiv',
    
    # Long-T5 (Guo et al., 2022)
    # Rationale: Efficient long-range attention
    'LongT5-Base': 'google/long-t5-tglobal-base',
    
    # Additional baselines for comparison
    'DistilBART': 'sshleifer/distilbart-cnn-12-6',
    'mT5-Base': 'google/mt5-base',
}

# ==============================================================================
# STEP 3: MULTI-DOCUMENT CONFIGURATION
# ==============================================================================
# Methodology Step 3: "Multiple related articles will be grouped together at
# the event level, concatenated to form a long-context input"

MULTI_DOC_CONFIG = {
    # Maximum articles to concatenate per event
    # Rationale: LED supports 16,384 tokens
    #            Avg article = 389 words ≈ 500 tokens
    #            5 articles ≈ 2,500 tokens (safe limit)
    'max_articles_per_event': 5,
    
    # Separator between documents
    # Rationale: Clear delimiter helps model distinguish sources
    'doc_separator': '\n\n[DOCUMENT]\n\n',
    
    # Concatenation order
    # Rationale: Chronological maintains event timeline (helps temporal consistency)
    'concat_order': 'chronological',
    
    # Whether to use multi-document mode
    # Set to True when event-grouped data is available
    'enable_multi_doc': False,  # Set to True when you have event groupings
}

# ==============================================================================
# EVALUATION CONFIGURATION
# ==============================================================================
EVAL_CONFIG = {
    # Test set size
    'test_samples': 200,  # Full test set for publication quality
    
    # Input truncation
    'max_article_length': 1024,  # Tokens (for most models)
    'max_article_length_led': 4096,  # For LED/Long-T5 (can handle longer)
    
    # Generation 9parameters
    'min_summary_length': 40,
    'max_summary_length': 150,
    'num_beams': 4,
    'do_sample': False,
    'early_stopping': True,
    
    # ROUGE configuration
    'rouge_metrics': ['rouge1', 'rouge2', 'rougeL'],
    'use_stemmer': True,
}

# ==============================================================================
# STEP 5: EVALUATION METRICS
# ==============================================================================
# Methodology Step 5: "ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, and QAGS"

EVALUATION_METRICS = {
    # ROUGE - lexical overlap (always enabled)
    'use_rouge': True,
    
    # BERTScore - semantic similarity
    # Rationale: Captures meaning beyond word overlap
    'use_bertscore': True,
    'bertscore_model': 'microsoft/deberta-xlarge-mnli',
    
    # QAGS - factual consistency (expensive, use on subset)
    # Rationale: Question-answering based factuality check
    'use_qags': True,
    'qags_samples': 500,  # Subset for computational efficiency
}

# ==============================================================================
# STEP 4: FACTUALITY MODULE CONFIGURATION
# ==============================================================================
# Methodology Step 4: "Factuality-aware post-processing module"
# Three checks: Entity, Temporal, Cross-document fact alignment

FACTUALITY_WEIGHTS = {
    # Entity Consistency (Named Entity Check)
    # Rationale: Entity errors most severe (wrong person/place/org)
    'entity': 0.40,
    
    # Temporal Consistency (Timeline Check)
    # Rationale: Date and event ordering critical for news
    'temporal': 0.30,
    
    # Semantic Consistency (Cross-document fact alignment)
    # Rationale: Measures overall content alignment
    'semantic': 0.30,
}

# Validate weights sum to 1.0
assert abs(sum(FACTUALITY_WEIGHTS.values()) - 1.0) < 0.001

# Entity types to verify
ENTITY_TYPES = ['PERSON', 'ORG', 'GPE', 'DATE']

# NLP tools
SPACY_MODEL = 'en_core_web_sm'
SENTENCE_ENCODER = 'all-MiniLM-L6-v2'
SEMANTIC_THRESHOLD = 0.5

# Date patterns for Indian English
DATE_PATTERNS = [
    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
    r'\d{4}',
    r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
    r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',
]

# ==============================================================================
# HARDWARE
# ==============================================================================
DEVICE = 0 if torch.cuda.is_available() else -1

# ==============================================================================
# LOGGING
# ==============================================================================
print(f"\n{'='*80}")
print(f" Configuration Loaded - Aligned with 5-Step Methodology")
print(f"{'='*80}")
print(f" Project: {PROJECT_INFO['title']}")
print(f" Author: {PROJECT_INFO['author']}")
print(f"{'='*80}")
print(f" Dataset: {PROJECT_INFO['dataset']}")
print(f"  Device: {'GPU' if DEVICE == 0 else 'CPU'}")
print(f" Test Samples: {EVAL_CONFIG['test_samples']:,}")
print(f" Baseline Models: {len(BASELINE_MODELS)}")
print(f"{'='*80}")
print(f"Methodology Alignment:")
print(f"  Step 1:  Source domain models (CNN/DM, XSum)")
print(f"  Step 2:  LED & Long-T5 baselines")
print(f"  Step 3:  Multi-doc concatenation ready")
print(f"  Step 4:  Factuality module configured")
print(f"  Step 5:  ROUGE + BERTScore + QAGS")
print(f"{'='*80}\n")

# Create directories
for path in RESULT_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)