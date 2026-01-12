"""
FA-LED: Factuality-Aware LED System
Date: January 2025
"""

from transformers import LEDTokenizer, LEDForConditionalGeneration
import torch
from factuality_module import FactualityChecker
from rouge_score import rouge_scorer
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime
import time
import spacy
import re


class EntityCorrector:
    """Correct entity hallucinations"""
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
    
    def correct(self, summary, source):
        """Remove entities not in source"""
        doc_summary = self.nlp(summary)
        doc_source = self.nlp(source)
        
        # Get entities
        summary_entities = {ent.text: ent.label_ for ent in doc_summary.ents}
        source_entities = {ent.text: ent.label_ for ent in doc_source.ents}
        
        corrected = summary
        for ent in summary_entities:
            # Check if entity in source
            if ent not in source_entities and ent.lower() not in source.lower():
                # Hallucinated - remove
                corrected = corrected.replace(ent, "")
        
        # Clean up extra spaces
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        return corrected


class TemporalAligner:
    """Fix temporal inconsistencies"""
    
    def align(self, summary, source):
        """Fix dates"""
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}'
        
        summary_dates = re.findall(date_pattern, summary)
        source_dates = re.findall(date_pattern, source)
        
        corrected = summary
        for date in summary_dates:
            if date not in source_dates:
                # Wrong date - remove
                corrected = corrected.replace(date, "")
        
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        return corrected


class FactualityAwareLED:
    """
    FA-LED: LED + Factuality Enhancement
    Novel Contribution
    """
    
    def __init__(self, device='cpu'):
        print("\n" + "="*80)
        print("FA-LED: FACTUALITY-AWARE LED")
        print("="*80)
        
        # Load LED
        print("\n Loading LED-ArXiv...")
        self.tokenizer = LEDTokenizer.from_pretrained('allenai/led-large-16384-arxiv')
        self.model = LEDForConditionalGeneration.from_pretrained('allenai/led-large-16384-arxiv')
        
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Factuality modules
        print(" Loading factuality modules...")
        self.entity_corrector = EntityCorrector()
        self.temporal_aligner = TemporalAligner()
        self.fact_checker = FactualityChecker()
        
        print("\n" + "="*80)
        print("FA-LED READY")
        print("="*80 + "\n")
    
    def generate_baseline(self, article):
        """LED without factuality (baseline)"""
        inputs = self.tokenizer(
            article,
            max_length=16384,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def generate_enhanced(self, article):
        """LED + Factuality (proposed)"""
        # Generate baseline
        raw = self.generate_baseline(article)
        
        # Apply factuality corrections
        entity_corrected = self.entity_corrector.correct(raw, article)
        final = self.temporal_aligner.align(entity_corrected, article)
        
        return final
    
    def evaluate_sample(self, article, reference):
        """Evaluate one sample: baseline vs enhanced"""
        
        # Generate both
        baseline = self.generate_baseline(article)
        enhanced = self.generate_enhanced(article)
        
        # ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        baseline_rouge = scorer.score(reference, baseline)
        enhanced_rouge = scorer.score(reference, enhanced)
        
        # Factuality scores
        baseline_fact = self.fact_checker.compute_factuality_score(baseline, article)
        enhanced_fact = self.fact_checker.compute_factuality_score(enhanced, article)
        
        return {
            'baseline': {
                'summary': baseline,
                'rouge1': baseline_rouge['rouge1'].fmeasure,
                'rouge2': baseline_rouge['rouge2'].fmeasure,
                'rougeL': baseline_rouge['rougeL'].fmeasure,
                'entity': baseline_fact['entity'],
                'temporal': baseline_fact['temporal'],
                'semantic': baseline_fact['semantic'],
                'factuality': baseline_fact['overall']
            },
            'enhanced': {
                'summary': enhanced,
                'rouge1': enhanced_rouge['rouge1'].fmeasure,
                'rouge2': enhanced_rouge['rouge2'].fmeasure,
                'rougeL': enhanced_rouge['rougeL'].fmeasure,
                'entity': enhanced_fact['entity'],
                'temporal': enhanced_fact['temporal'],
                'semantic': enhanced_fact['semantic'],
                'factuality': enhanced_fact['overall']
            }
        }


def evaluate_faled(num_samples=200):
    """
    Evaluate FA-LED on NewsSumm
    Compare baseline LED vs FA-LED (enhanced)
    """
    print("\n" + "="*80)
    print("FA-LED EVALUATION")
    print("="*80)
    
    # Load data
    test_df = pd.read_csv('data/processed/test_full.csv').head(num_samples)
    print(f"\n Loaded {len(test_df)} samples")
    
    # Initialize
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f" Device: {device}")
    
    faled = FactualityAwareLED(device=device)
    
    # Evaluate
    baseline_results = []
    enhanced_results = []
    
    print("\n Evaluating...\n")
    start_time = time.time()
    
    for idx in tqdm(range(len(test_df)), desc="FA-LED Evaluation"):
        try:
            row = test_df.iloc[idx]
            article = str(row['article'])
            reference = str(row['summary'])
            
            result = faled.evaluate_sample(article, reference)
            
            baseline_results.append({
                'sample_id': idx,
                **result['baseline']
            })
            
            enhanced_results.append({
                'sample_id': idx,
                **result['enhanced']
            })
            
        except Exception as e:
            print(f"\n✗ Error on {idx}: {str(e)[:100]}")
            continue
    
    elapsed = time.time() - start_time
    
    # Aggregate baseline
    baseline_aggregate = {
        'model_name': 'LED-ArXiv (Baseline)',
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': num_samples,
        'successful_samples': len(baseline_results),
        'success_rate': round(len(baseline_results) / num_samples * 100, 2),
        
        'rouge1_mean': round(np.mean([r['rouge1'] for r in baseline_results]) * 100, 2),
        'rouge2_mean': round(np.mean([r['rouge2'] for r in baseline_results]) * 100, 2),
        'rougeL_mean': round(np.mean([r['rougeL'] for r in baseline_results]) * 100, 2),
        
        'entity_mean': round(np.mean([r['entity'] for r in baseline_results]), 3),
        'temporal_mean': round(np.mean([r['temporal'] for r in baseline_results]), 3),
        'semantic_mean': round(np.mean([r['semantic'] for r in baseline_results]), 3),
        'factuality_mean': round(np.mean([r['factuality'] for r in baseline_results]), 3),
        
        'evaluation_time_hours': round(elapsed / 3600, 2)
    }
    
    # Aggregate enhanced
    enhanced_aggregate = {
        'model_name': 'FA-LED (Proposed)',
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': num_samples,
        'successful_samples': len(enhanced_results),
        'success_rate': round(len(enhanced_results) / num_samples * 100, 2),
        
        'rouge1_mean': round(np.mean([r['rouge1'] for r in enhanced_results]) * 100, 2),
        'rouge2_mean': round(np.mean([r['rouge2'] for r in enhanced_results]) * 100, 2),
        'rougeL_mean': round(np.mean([r['rougeL'] for r in enhanced_results]) * 100, 2),
        
        'entity_mean': round(np.mean([r['entity'] for r in enhanced_results]), 3),
        'temporal_mean': round(np.mean([r['temporal'] for r in enhanced_results]), 3),
        'semantic_mean': round(np.mean([r['semantic'] for r in enhanced_results]), 3),
        'factuality_mean': round(np.mean([r['factuality'] for r in enhanced_results]), 3),
        
        'evaluation_time_hours': round(elapsed / 3600, 2)
    }
    
    # Print comparison
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    print(f"\nLED-ArXiv (Baseline):")
    print(f"  ROUGE-2: {baseline_aggregate['rouge2_mean']:.2f}%")
    print(f"  Entity: {baseline_aggregate['entity_mean']:.3f}")
    print(f"  Temporal: {baseline_aggregate['temporal_mean']:.3f}")
    print(f"  Factuality: {baseline_aggregate['factuality_mean']:.3f}")
    
    print(f"\nFA-LED (Proposed):")
    print(f"  ROUGE-2: {enhanced_aggregate['rouge2_mean']:.2f}%")
    print(f"  Entity: {enhanced_aggregate['entity_mean']:.3f}")
    print(f"  Temporal: {enhanced_aggregate['temporal_mean']:.3f}")
    print(f"  Factuality: {enhanced_aggregate['factuality_mean']:.3f}")
    
    print(f"\nImprovement:")
    print(f"  ROUGE-2: {enhanced_aggregate['rouge2_mean'] - baseline_aggregate['rouge2_mean']:+.2f}%")
    print(f"  Factuality: {enhanced_aggregate['factuality_mean'] - baseline_aggregate['factuality_mean']:+.3f}")
    
    print(f"\nTime: {elapsed/3600:.2f} hours")
    print("="*80 + "\n")
    
    # Save results
    output_dir = Path('results/proposed_model')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'led_baseline_aggregate.json', 'w') as f:
        json.dump(baseline_aggregate, f, indent=2)
    
    with open(output_dir / 'faled_aggregate.json', 'w') as f:
        json.dump(enhanced_aggregate, f, indent=2)
    
    with open(output_dir / 'led_baseline_detailed.json', 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    with open(output_dir / 'faled_detailed.json', 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    
    print(f" Results saved to: {output_dir}\n")
    
    return baseline_aggregate, enhanced_aggregate


if __name__ == "__main__":
    # Quick test
    print("Testing FA-LED...")
    
    article = """PASIGHAT, 4 Jan: Forest officials from Daporijo,led by DFO (T) Boken Pao, 
    recently seized large quantities of wild animal carcasses and wild meat from the 
    markets in and around Daporijo town in Upper Subansiri district."""
    
    reference = """Forest officials seized wild animal carcasses in Daporijo on January 4."""
    
    faled = FactualityAwareLED()
    result = faled.evaluate_sample(article, reference)
    
    print(f"\nBaseline: {result['baseline']['summary'][:100]}...")
    print(f"Enhanced: {result['enhanced']['summary'][:100]}...")
    print(f"\nFactuality improvement: {result['enhanced']['factuality'] - result['baseline']['factuality']:+.3f}")
    print("\n FA-LED working!")