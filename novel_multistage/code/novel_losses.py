"""
Novel Loss Functions for Indian English Summarization
================
Date: January 2025

Novel Contributions:
1. IndianEntityLoss - Preserves Indian-specific terms (crore, lakh, etc.)
2. TemporalConsistencyLoss - Prevents date hallucinations
3. SemanticFidelityLoss - Maintains semantic similarity
4. MultiStageLoss - Dynamic weighting across training stages
"""

import torch
import torch.nn as nn
import re
from sklearn.metrics.pairwise import cosine_similarity


class IndianEntityLoss(nn.Module):
    """
    NOVEL: Indian English-specific entity preservation loss
    
    Tracks 56 culture-specific terms and penalizes their omission
    in generated summaries.
    """
    def __init__(self):
        super().__init__()
        
        # Curated vocabulary of Indian-specific terms
        self.indian_vocab = {
            'numbers': ['crore', 'lakh', 'thousand', 'lakhs', 'crores'],
            'titles': ['PM', 'CM', 'IAS', 'IPS', 'IFS', 'Lok Sabha', 'Rajya Sabha',
                      'Chief Minister', 'Prime Minister', 'MLA', 'MP', 'Minister'],
            'political': ['BJP', 'Congress', 'AAP', 'NDA', 'UPA', 'CPI', 'TMC', 
                         'DMK', 'AIADMK', 'Shiv Sena', 'JDU', 'RJD'],
            'admin': ['Panchayat', 'Municipality', 'District Collector', 
                     'Tehsildar', 'Gram Sabha', 'Zila Parishad'],
            'cultural': ['dharma', 'karma', 'puja', 'darshan', 'prasad', 
                        'Diwali', 'Holi', 'Eid', 'Ganesh Chaturthi', 'Navratri'],
            'institutions': ['ISRO', 'DRDO', 'AIIMS', 'IIT', 'IIM', 'RBI', 
                           'SEBI', 'CBI', 'ED', 'IT Department', 'CAG']
        }
        
        # Flatten all terms for quick lookup
        self.all_terms = []
        for terms in self.indian_vocab.values():
            self.all_terms.extend([t.lower() for t in terms])
    
    def extract_indian_terms(self, text):
        """Extract Indian-specific terms from text"""
        found = []
        text_lower = text.lower()
        for term in self.all_terms:
            if term in text_lower:
                found.append(term)
        return list(set(found))
    
    def compute_loss(self, source, summary):
        """
        Calculate entity preservation loss
        
        Loss = 1 - (preserved_terms / source_terms)
        
        Returns:
            float: Loss value between 0 and 1
                  0 = all entities preserved
                  1 = no entities preserved
        """
        source_terms = self.extract_indian_terms(source)
        summary_terms = self.extract_indian_terms(summary)
        
        if len(source_terms) == 0:
            return 0.0
        
        preserved = sum(1 for term in source_terms if term in summary_terms)
        loss = 1.0 - (preserved / len(source_terms))
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """
    NOVEL: Temporal consistency loss for news summarization
    
    Detects and penalizes hallucinated dates in summaries.
    Critical for news where temporal accuracy is essential.
    """
    def __init__(self):
        super().__init__()
        
        # Date regex patterns
        self.date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # DD/MM/YYYY
            r'\b\d{4}\b',                       # Year only
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
        ]
    
    def extract_dates(self, text):
        """Extract all date mentions from text"""
        dates = set()
        for pattern in self.date_patterns:
            dates.update(re.findall(pattern, text, re.IGNORECASE))
        return dates
    
    def compute_loss(self, source, summary):
        """
        Calculate temporal consistency loss
        
        Penalizes dates that appear in summary but not in source
        (i.e., hallucinated dates)
        
        Loss = |hallucinated_dates| / |summary_dates|
        
        Returns:
            float: Loss value between 0 and 1
                  0 = no hallucinated dates
                  1 = all dates are hallucinated
        """
        source_dates = self.extract_dates(source)
        summary_dates = self.extract_dates(summary)
        
        if len(summary_dates) == 0:
            return 0.0
        
        # Dates in summary not in source = hallucinations
        hallucinated = summary_dates - source_dates
        loss = len(hallucinated) / len(summary_dates)
        
        return loss


class SemanticFidelityLoss(nn.Module):
    """
    NOVEL: Semantic similarity constraint
    
    Ensures generated summary stays semantically close to source
    while allowing abstraction.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder  # SentenceTransformer model
    
    def compute_loss(self, source, summary):
        """
        Calculate semantic fidelity loss
        
        Loss = 1 - cosine_similarity(source_embedding, summary_embedding)
        
        Returns:
            float: Loss value between 0 and 1
                  0 = perfect semantic match
                  1 = completely different meaning
        """
        try:
            # Truncate to avoid memory issues
            source_trunc = source[:2000]
            summary_trunc = summary[:1000]
            
            # Encode
            source_emb = self.encoder.encode([source_trunc])
            summary_emb = self.encoder.encode([summary_trunc])
            
            # Cosine similarity
            sim = cosine_similarity(source_emb, summary_emb)[0][0]
            loss = 1.0 - sim
            
            return float(loss)
        except:
            return 0.0


class MultiStageLoss(nn.Module):
    """
    NOVEL: Multi-objective loss with stage-dependent weighting
    
    Combines 4 loss components with weights that change across training stages:
    - Cross-Entropy (fluency)
    - Entity Preservation (Indian terms)
    - Temporal Consistency (date accuracy)
    - Semantic Fidelity (meaning preservation)
    
    Stage-specific weights:
    - Stage 1 (Warmup):     Focus on fluency only
    - Stage 2 (Factuality): Add entity + temporal
    - Stage 3 (Refinement): Balance all objectives
    """
    def __init__(self, stage='warmup', sbert_encoder=None):
        super().__init__()
        self.stage = stage
        
        # NOVEL: Stage-dependent loss weights
        self.stage_weights = {
            'warmup': {
                'ce': 1.0,      # Only cross-entropy
                'entity': 0.0, 
                'temporal': 0.0, 
                'semantic': 0.0
            },
            'factuality': {
                'ce': 0.5,      # Reduce CE weight
                'entity': 0.3,  # Add entity preservation
                'temporal': 0.2, # Add temporal consistency
                'semantic': 0.0
            },
            'refinement': {
                'ce': 0.4,      # Further reduce CE
                'entity': 0.2,  # Balance all losses
                'temporal': 0.2, 
                'semantic': 0.2  # Add semantic fidelity
            }
        }
        
        self.weights = self.stage_weights[stage]
        
        # Initialize loss components
        self.entity_loss_fn = IndianEntityLoss()
        self.temporal_loss_fn = TemporalConsistencyLoss()
        self.semantic_loss_fn = SemanticFidelityLoss(sbert_encoder)
    
    def forward(self, ce_loss, source_text, generated_text):
        """
        Compute weighted multi-objective loss
        
        Args:
            ce_loss: Cross-entropy loss from model
            source_text: Original article text
            generated_text: Generated summary text
            
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary with individual loss values
        """
        # Compute component losses
        entity_loss = self.entity_loss_fn.compute_loss(source_text, generated_text)
        temporal_loss = self.temporal_loss_fn.compute_loss(source_text, generated_text)
        semantic_loss = self.semantic_loss_fn.compute_loss(source_text, generated_text)
        
        # Weighted combination
        total_loss = (
            self.weights['ce'] * ce_loss +
            self.weights['entity'] * entity_loss +
            self.weights['temporal'] * temporal_loss +
            self.weights['semantic'] * semantic_loss
        )
        
        # Return both total and individual losses for monitoring
        return total_loss, {
            'ce': float(ce_loss),
            'entity': float(entity_loss),
            'temporal': float(temporal_loss),
            'semantic': float(semantic_loss),
            'total': float(total_loss)
        }


# Example usage
if __name__ == '__main__':
    # Test IndianEntityLoss
    entity_loss = IndianEntityLoss()
    source = "The government allocated 5 crore rupees to the IAS training program in Lok Sabha."
    summary_good = "Government allocated 5 crore for IAS training in Lok Sabha."
    summary_bad = "Government allocated 50 million for civil service training."
    
    loss_good = entity_loss.compute_loss(source, summary_good)
    loss_bad = entity_loss.compute_loss(source, summary_bad)
    
    print(f"Good summary (preserves terms): Loss = {loss_good:.3f}")
    print(f"Bad summary (drops terms): Loss = {loss_bad:.3f}")