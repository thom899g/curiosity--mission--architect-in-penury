"""
Reality Distillation Engine: Cross-validation and anomaly detection
First line of defense against corrupted or manipulated data
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any
import hashlib
import requests
import time
from sklearn.ensemble import IsolationForest
from scipy import stats
import firebase_admin
from firebase_admin import firestore
import json

logger = logging.getLogger(__name__)

class RealityDistiller:
    """Ingests and validates market data from multiple sources"""
    
    def __init__(self, firestore_client=None, min_sources: int = 3):
        """
        Initialize distiller with cross-validation requirements.
        
        Args:
            firestore_client: Firestore client (optional)
            min_sources: Minimum independent sources for validation
        """
        self.db = firestore_client
        self.min_sources = min_sources
        self.cross_reference_window_hours = 1.0
        
        # Initialize anomaly detector
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        # Cache for recent data to avoid excessive Firestore queries
        self.data_cache: Dict[str, List[Dict]] = {}
        self.cache_ttl = timedelta(minutes=5)
        self.cache_timestamps: Dict[str, datetime] = {}
        
        logger.info(f"RealityDistiller initialized (min_sources={min_sources})")
    
    def ingest_market_datum(self, raw_data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Process raw market data with validation and anomaly detection.
        
        Args:
            raw_data: {
                'symbol': str,
                'price': float,
                'volume': float,
                'timestamp': datetime or str,
                'exchange': str
            }
            source: Source identifier
        
        Returns:
            Dict with processed data and confidence metrics
        """
        start_time = time.time()
        
        try:
            # Validate input structure
            required_keys = ['symbol', 'price', 'timestamp']
            for key in required_keys:
                if key not in raw_data:
                    logger.error(f"Missing required key: {key}")
                    raise ValueError(f"Missing required key: {key}")
            
            symbol = raw_data['symbol']
            price = float(raw_data['price'])
            
            # Convert timestamp to datetime if string
            if isinstance(raw_data['timestamp'], str):
                timestamp = datetime.fromisoformat(raw_data['timestamp'].replace('Z', '+00:00'))
            else:
                timestamp = raw_data['timestamp']
            
            # Source fingerprinting
            source_hash = hashlib.sha256(source.encode()).hexdigest()[:16]
            
            # Cross-reference with other sources
            recent_data = self._query_parallel_sources(symbol, timestamp)
            
            confidence = 0.5  # Default prior
            anomaly_score = 0.0
            
            if len(recent_data) >= self.min_sources:
                # Calculate consensus metrics
                prices = [d['price'] for d in recent_data]
                
                # Robust statistics (median absolute deviation)
                median_price = np.median(prices)
                mad = stats.median_abs_deviation(prices, scale='normal')
                
                # Calculate anomaly score
                if mad > 0:
                    anomaly_score = abs(price - median_price) / mad
                else:
                    anomaly_score = abs(price - median_price) / (abs(median_price) * 0.01 + 1e-6)
                
                # Update source reliability
                source_reliability = self._update_source_reliability(
                    source_hash, anomaly_score, len(recent_data)
                )
                
                # Confidence based on reliability and anomaly
                confidence = source_reliability * np.exp(-anomaly_score)
                confidence = max(0.1, min(0.99,