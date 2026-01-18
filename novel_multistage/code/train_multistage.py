"""
Multi-Stage Training Script for Indian English Summarization
============================================================
Date: January 2025

Novel Contribution: 3-stage progressive training with dynamic loss weighting

Usage:
    python train_multistage.py --train_data data/train_2k.csv \
                           --val_data data/val_1k.csv \
                             --output_dir checkpoints/
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LEDForConditionalGeneration,
    LEDTokenizer,
    get_linear_schedule_with_warmup
)
import pandas as pd
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path

# Import our novel losses
from novel_losses import MultiStageLoss


class NewsDataset(Dataset):
    """Dataset for Indian English news articles"""
    
    def __init__(self, csv_path, tokenizer, max_length=1024):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Tokenize article
        inputs = self.tokenizer(
            str(row['article']),
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize summary
        targets = self.tokenizer(
            str(row['summary']),
            max_length=256,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
            'source_text': str(row['article']),
            'target_text': str(row['summary'])
        }


def train_stage(model, tokenizer, dataloader, optimizer, scheduler,
                stage_name, device, sbert_encoder, epochs=1):
    """
    Train one stage of the 3-stage pipeline
    
    Args:
        stage_name: 'warmup', 'factuality', or 'refinement'
    """
    print(f"\n{'='*70}")
    print(f"STAGE: {stage_name.upper()}")
    print(f"{'='*70}")
    
    model.train()
    model.to(device)
    
    # Initialize stage-specific loss
    loss_fn = MultiStageLoss(stage=stage_name, sbert_encoder=sbert_encoder)
    stage_history = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            source_text = batch['source_text'][0]
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            ce_loss = outputs.loss
            
            # Compute factuality losses (every 10 batches)
            if batch_idx % 10 == 0 and stage_name != 'warmup':
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids,
                        max_length=256,
                        num_beams=2,
                        early_stopping=True
                    )
                    generated_text = tokenizer.decode(
                        generated_ids[0],
                        skip_special_tokens=True
                    )
                
                total_loss, loss_dict = loss_fn(ce_loss, source_text, generated_text)
            else:
                total_loss = ce_loss
                loss_dict = {
                    'ce': float(ce_loss),
                    'entity': 0.0,
                    'temporal': 0.0,
                    'semantic': 0.0,
                    'total': float(ce_loss)
                }
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss_dict)
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'entity': f"{loss_dict['entity']:.3f}"
            })
            
            # Free memory
            del input_ids, attention_mask, labels, outputs
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Compute epoch averages
        avg_losses = {
            k: np.mean([d[k] for d in epoch_losses])
            for k in epoch_losses[0].keys()
        }
        
        print(f"\nEpoch {epoch+1} Results:")
        for k, v in avg_losses.items():
            print(f"  {k}: {v:.4f}")
        
        stage_history.append(avg_losses)
    
    return stage_history


def main(args):
    """Main training function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model and tokenizer
    print("\nLoading LED model...")
    model_name = 'allenai/led-large-16384-arxiv'
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    model = LEDForConditionalGeneration.from_pretrained(model_name)
    
    # Load data
    print("\nLoading data...")
    train_dataset = NewsDataset(args.train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * 3  # 3 stages
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )
    
    # Load SBERT for semantic loss
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Training history
    training_history = {}
    
    # Stage 1: Warmup
    stage1_history = train_stage(
        model, tokenizer, train_loader, optimizer, scheduler,
        'warmup', device, sbert, epochs=1
    )
    training_history['stage1_warmup'] = stage1_history
    model.save_pretrained(f'{args.output_dir}/stage1')
    
    # Stage 2: Factuality
    stage2_history = train_stage(
        model, tokenizer, train_loader, optimizer, scheduler,
        'factuality', device, sbert, epochs=1
    )
    training_history['stage2_factuality'] = stage2_history
    model.save_pretrained(f'{args.output_dir}/stage2')
    
    # Stage 3: Refinement
    stage3_history = train_stage(
        model, tokenizer, train_loader, optimizer, scheduler,
        'refinement', device, sbert, epochs=1
    )
    training_history['stage3_refinement'] = stage3_history
    
    # Save final model
    model.save_pretrained(f'{args.output_dir}/final_model')
    tokenizer.save_pretrained(f'{args.output_dir}/final_model')
    
    # Save training history
    with open(f'{args.output_dir}/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\n Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--lr', type=float, default=3e-5)
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)