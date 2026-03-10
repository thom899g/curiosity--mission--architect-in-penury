"""
Probabilistic Prophet Core: Belief State Engine
Bayesian network with epistemic uncertainty tracking
"""
import numpy as np
import pandas as pd
from scipy import stats
import firebase_admin
from firebase_admin import firestore, credentials
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProbabilisticState:
    """
    Bayesian belief state engine with immutable constitutional rules.
    Every belief carries epistemic uncertainty and source reliability tracking.
    """
    
    def __init__(self, constitution_path: str = "constitution.json"):
        """
        Initialize belief state with constitutional constraints.
        
        Args:
            constitution_path: Path to immutable constitution JSON
        """
        try:
            with open(constitution_path, 'r') as f:
                self.constitution = json.load(f)
            logger.info(f"Loaded constitution v{self.constitution['version']}")
        except FileNotFoundError:
            logger.error(f"Constitution file {constitution_path} not found")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in constitution file")
            raise
        
        # Initialize Firestore if not already initialized
        if not firebase_admin._apps:
            try:
                cred = credentials.Certificate('serviceAccountKey.json')
                firebase_admin.initialize_app(cred)
                logger.info("Firebase Admin initialized")
            except FileNotFoundError:
                logger.warning("Firebase service account key not found, using mock mode")
                self.db = None
            except Exception as e:
                logger.error(f"Firebase initialization failed: {e}")
                self.db = None
        else:
            self.db = firestore.client()
        
        # Core belief state
        self.beliefs: Dict[str, Dict] = {}  # variable -> Gaussian mixture model
        self.epistemic_confidence = 0.95
        self.last_updated = datetime.utcnow()
        self.belief_update_count = 0
        
        # Track source reliability
        self.source_reliability: Dict[str, float] = {}
        
        logger.info("ProbabilisticState initialized")
    
    def update_belief(self, variable: str, evidence: Dict[str, Any]) -> bool:
        """
        Bayesian belief update with source reliability weighting.
        
        Args:
            variable: Belief variable name (e.g., "BTC_volatility_1h")
            evidence: {
                'mean': float,
                'std': float,
                'source_confidence': float (0-1),
                'source_id': str,
                'timestamp': datetime
            }
        
        Returns:
            bool: True if belief was updated, False if rejected
        """
        try:
            # Constitutional check: minimum evidence confidence
            min_conf = self.constitution['immutable_rules']['min_evidence_confidence']
            if evidence['source_confidence'] < min_conf:
                logger.warning(
                    f"Evidence for {variable} rejected: confidence {evidence['source_confidence']:.3f} "
                    f"< minimum {min_conf}"
                )
                return False
            
            # Apply source reliability adjustment if available
            source_id = evidence.get('source_id', 'unknown')
            if source_id in self.source_reliability:
                adjusted_confidence = evidence['source_confidence'] * self.source_reliability[source_id]
                evidence['source_confidence'] = max(min_conf, adjusted_confidence)
                logger.debug(f"Adjusted confidence for {source_id}: {adjusted_confidence:.3f}")
            
            # Initialize variable if first time
            if variable not in self.beliefs:
                self.beliefs[variable] = {
                    'means': [],
                    'stds': [],
                    'weights': [],
                    'timestamps': [],
                    'sources': []
                }
            
            # Calculate component weight based on recency and confidence
            recency_decay = np.exp(
                -0.1 * (datetime.utcnow() - evidence['timestamp']).total_seconds() / 3600
            )
            component_weight = evidence['source_confidence'] * recency_decay
            
            # Add new component
            self.beliefs[variable]['means'].append(evidence['mean'])
            self.beliefs[variable]['stds'].append(max(1e-6, evidence['std']))  # Avoid zero std
            self.beliefs[variable]['weights'].append(component_weight)
            self.beliefs[variable]['timestamps'].append(evidence['timestamp'])
            self.beliefs[variable]['sources'].append(source_id)
            
            # Normalize weights to sum to 1
            total_weight = sum(self.beliefs[variable]['weights'])
            if total_weight > 0:
                self.beliefs[variable]['weights'] = [w/total_weight 
                                                   for w in self.beliefs[variable]['weights']]
            
            # Prune old components (keep only top 10 by weight)
            if len(self.beliefs[variable]['weights']) > 10:
                indices = np.argsort(self.beliefs[variable]['weights'])[-10:]
                for key in ['means', 'stds', 'weights', 'timestamps', 'sources']:
                    self.beliefs[variable][key] = [
                        self.beliefs[variable][key][i] for i in indices
                    ]
            
            self.last_updated = datetime.utcnow()
            self.belief_update_count += 1
            
            # Persist to Firestore if available
            if self.db:
                try:
                    doc_ref = self.db.collection('belief_states').document(variable)
                    doc_ref.set({
                        'variable': variable,
                        'components': self.beliefs[variable],
                        'last_updated': self.last_updated,
                        'update_count': self.belief_update_count
                    }, merge=True)
                    logger.debug(f"Persisted belief state for {variable}")
                except Exception as e:
                    logger.error(f"Failed to persist belief state: {e}")
            
            logger.info(f"Updated belief for {variable} with {len(self.beliefs[variable]['means'])} components")
            return True
            
        except KeyError as e:
            logger.error(f"Missing key in evidence: {e}")
            return False
        except Exception as e:
            logger.error(f"Belief update failed: {e}")
            return False
    
    def query(self, variable: str, confidence_interval: float = 0.95) -> Tuple[float, float, float, float]:
        """
        Query belief distribution with uncertainty quantification.
        
        Args:
            variable: Belief variable name
            confidence_interval: Desired confidence interval (0-1)
        
        Returns:
            Tuple: (lower_bound, mean, upper_bound, epistemic_uncertainty)
            Returns (0, 0, 0, 1) if variable unknown
        """
        if variable not in self.beliefs or not self.beliefs[variable]['weights']:
            logger.warning(f"Variable {variable} not in belief state")
            return (0.0, 0.0, 0.0, 1.0)
        
        try:
            means = np.array(self.beliefs[variable]['means'])
            stds = np.array(self.beliefs[variable]['stds'])
            weights = np.array(self.beliefs[variable]['weights'])
            
            # Generate weighted samples
            all_samples = []
            for mean, std, weight in zip(means, stds, weights):
                n_samples = max(1, int(10000 * weight))
                samples = np.random.normal(mean, std, n_samples)
                all_samples.extend(samples)
            
            if not all_samples:
                return (0.0, 0.0, 0.0, 1.0)
            
            all_samples = np.array(all_samples)
            
            # Calculate statistics
            mean_val = np.mean(all_samples)
            lower = np.percentile(all_samples, (1 - confidence_interval) * 50)
            upper = np.percentile(all_samples, confidence_interval * 50 + 50)
            
            # Epistemic uncertainty = coefficient of variation of component means
            if len(means) > 1:
                weighted_mean = np.average(means, weights=weights)
                weighted_std = np.sqrt(np.average((means - weighted_mean)**2, weights=weights))
                epistemic = weighted_std / (abs(weighted_mean) + 1e-6)
            else:
                epistemic = 1.0  # Maximum uncertainty with single component
            
            logger.debug(f"Query {variable}: {lower:.4f} < {mean_val:.4f} < {upper:.4f}, ε={epistemic:.3f}")
            return (float(lower), float(mean_val), float(upper), float(epistemic))
            
        except Exception as e:
            logger.error(f"Query failed for {variable}: {e}")
            return (0.0, 0.0, 0.0, 1.0)
    
    def update_source_reliability(self, source_id: str, success: bool, impact: float = 1.0):
        """
        Update source reliability based on prediction accuracy.
        
        Args:
            source_id: Unique source identifier
            success: Whether prediction was accurate
            impact: Magnitude of the prediction (for weighting)
        """
        current = self.source_reliability.get(source_id, 0.5)
        
        # Bayesian update of Beta distribution parameters
        alpha, beta = current * 10, (1 - current) * 10
        if success:
            alpha += impact
        else:
            beta += impact
        
        new_reliability = alpha / (alpha + beta)
        self.source_reliability[source_id] = max(0.1, min(0.99, new_reliability))
        
        logger.info(f"Source {source_id} reliability: {current:.3f} → {new_reliability:.3f}")
        
        # Persist to Firestore
        if self.db:
            try:
                self.db.collection('source_reliability').document(source_id).set({
                    'reliability': new_reliability,
                    'last_updated': datetime.utcnow(),
                    'sample_count': alpha + beta - 10  # Subtract prior
                }, merge=True)
            except Exception as e:
                logger.error(f"Failed to persist source reliability: {e}")