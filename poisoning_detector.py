#!/usr/bin/env python3
"""
Conservative Poisoning Detection System
======================================

Prioritizes high-confidence detection to reduce false positives in 2-client scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial.distance import cosine
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger("PoisoningDetector")

class PoisoningDetector:
    """
    Conservative poisoning detection system.
    """
    def __init__(self, 
                 similarity_threshold: float = 0.1,      # Very strict
                 outlier_threshold: float = 3.0,         # Higher threshold
                 reputation_threshold: float = 0.5,      # More lenient
                 reputation_decay: float = 0.95,         # Slower decay
                 save_logs: bool = True,
                 log_dir: str = "security_logs"):
        """
        Initialize conservative detection system.
        """
        self.similarity_threshold = similarity_threshold
        self.outlier_threshold = outlier_threshold
        self.reputation_threshold = reputation_threshold
        self.reputation_decay = reputation_decay
        
        self.client_reputations = {}
        self.detection_history = []
        self.round_count = 0
        
        self.save_logs = save_logs
        self.log_dir = log_dir
        if save_logs:
            os.makedirs(log_dir, exist_ok=True)
        
        print(f"🔧 CONSERVATIVE POISONING DETECTOR")
        print(f"   Similarity threshold: {similarity_threshold}")
        print(f"   Outlier threshold: {outlier_threshold}")
        print(f"   Reputation threshold: {reputation_threshold}")
    
    def analyze_round(self, 
                     client_updates: Dict[str, List[np.ndarray]], 
                     client_metrics: Optional[Dict[str, Dict[str, float]]] = None,
                     round_num: Optional[int] = None) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Conservative analysis prioritizing high-confidence detections.
        """
        if round_num is None:
            self.round_count += 1
            round_num = self.round_count
        else:
            self.round_count = round_num
        
        num_clients = len(client_updates)
        print(f"\n🔍 CONSERVATIVE DETECTION - ROUND {round_num}")
        print(f"Analyzing {num_clients} clients: {list(client_updates.keys())}")
        
        # Initialize reputations for new clients
        for client_id in client_updates.keys():
            if client_id not in self.client_reputations:
                self.client_reputations[client_id] = 1.0
        
        # Apply gentle reputation decay
        for client_id in self.client_reputations:
            old_rep = self.client_reputations[client_id]
            self.client_reputations[client_id] = old_rep * self.reputation_decay
        
        suspicious_clients = set()
        high_confidence_detections = {}
        
        # Method 1: High-Confidence Performance Detection (Most Reliable)
        print("--- High-Confidence Performance Detection ---")
        if client_metrics:
            perf_detections = self._detect_high_confidence_performance(client_metrics)
            suspicious_clients.update(perf_detections.keys())
            high_confidence_detections.update(perf_detections)
        
        # Method 2: Extreme Magnitude Detection (Conservative)
        print("--- Extreme Magnitude Detection ---")
        mag_detections = self._detect_extreme_magnitudes(client_updates)
        suspicious_clients.update(mag_detections.keys())
        high_confidence_detections.update(mag_detections)
        
        # Method 3: Pattern Detection (Only Extreme Cases)
        print("--- Extreme Pattern Detection ---")
        pattern_detections = self._detect_extreme_patterns(client_updates)
        for client_id in pattern_detections:
            suspicious_clients.add(client_id)
            high_confidence_detections[client_id] = high_confidence_detections.get(client_id, 0) + 0.8
        
        # Conservative filtering: Only keep very high confidence detections
        print("--- Conservative Validation ---")
        validated_suspicious = self._conservative_validation(
            suspicious_clients, high_confidence_detections, num_clients, client_metrics
        )
        
        # Update reputations (more conservative)
        for client_id in client_updates.keys():
            old_rep = self.client_reputations.get(client_id, 1.0)
            confidence = high_confidence_detections.get(client_id, 0)
            
            if client_id in validated_suspicious:
                # Confidence-based penalty (but more conservative)
                penalty = min(0.4, confidence * 0.3)  # Max 40% penalty
                new_rep = old_rep * (1 - penalty)
                self.client_reputations[client_id] = new_rep
                print(f"🚨 PENALIZED {client_id}: {old_rep:.3f} -> {new_rep:.3f} (conf: {confidence:.2f})")
            else:
                # Small reward for good behavior
                new_rep = min(1.0, old_rep + 0.02)
                self.client_reputations[client_id] = new_rep
                print(f"✅ REWARDED {client_id}: {old_rep:.3f} -> {new_rep:.3f}")
        
        # Determine trusted clients
        trusted_clients = [
            cid for cid, rep in self.client_reputations.items()
            if rep >= self.reputation_threshold and cid not in validated_suspicious
        ]
        
        # Conservative threat assessment
        threat_level = self._assess_threat_conservative(validated_suspicious, high_confidence_detections, num_clients)
        is_safe = threat_level in ["low", "medium"]
        
        detection_details = {
            "round": round_num,
            "threat_level": threat_level,
            "suspicious_clients": list(validated_suspicious),
            "trusted_clients": trusted_clients,
            "client_reputations": self.client_reputations.copy(),
            "high_confidence_detections": high_confidence_detections,
            "total_clients": num_clients
        }
        
        self._log_detection_results(detection_details)
        self.detection_history.append(detection_details)
        if self.save_logs:
            self._save_detection_log(detection_details)
        
        print(f"--- CONSERVATIVE RESULTS ---")
        print(f"  Suspicious: {list(validated_suspicious)}")
        print(f"  Trusted: {trusted_clients}")
        print(f"  Threat level: {threat_level}")
        
        return is_safe, trusted_clients, detection_details
    
    def _detect_high_confidence_performance(self, client_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Detect clients with very high confidence based on performance."""
        detections = {}
        
        print("  High-confidence performance analysis:")
        for client_id, metrics in client_metrics.items():
            accuracy = metrics.get('accuracy', 0)
            loss = metrics.get('loss', float('inf'))
            poisoning_active = metrics.get('poisoning_active', False)
            
            print(f"    {client_id}: acc={accuracy:.3f}, loss={loss:.3f}, poisoning={poisoning_active}")
            
            confidence = 0.0
            reasons = []
            
            # Extremely high confidence indicators
            if poisoning_active:
                confidence += 0.95  # Almost certain
                reasons.append("explicit poisoning indicator")
            
            if accuracy < 0.1:  # Extremely low accuracy
                confidence += 0.8
                reasons.append(f"extremely low accuracy ({accuracy:.3f})")
            
            if loss > 10.0:  # Very high loss
                confidence += 0.7
                reasons.append(f"extremely high loss ({loss:.3f})")
            
            # Combined poor performance (lower confidence)
            if accuracy < 0.3 and loss > 3.0:
                confidence += 0.4
                reasons.append("combined poor performance")
            
            # Only flag with very high confidence
            if confidence > 0.8:
                detections[client_id] = confidence
                print(f"    🚨 HIGH-CONFIDENCE FLAG: {client_id}: {', '.join(reasons)} (conf: {confidence:.2f})")
        
        return detections
    
    def _detect_extreme_magnitudes(self, client_updates: Dict[str, List[np.ndarray]]) -> Dict[str, float]:
        """Detect only extreme magnitude outliers."""
        if len(client_updates) < 2:
            return {}
        
        # Calculate magnitudes
        magnitudes = {}
        for client_id, update_list in client_updates.items():
            total_norm = sum(np.linalg.norm(param) for param in update_list if isinstance(param, np.ndarray))
            magnitudes[client_id] = total_norm
            print(f"  {client_id}: magnitude = {total_norm:.6f}")
        
        detections = {}
        values = list(magnitudes.values())
        
        if len(values) == 2:
            # For 2 clients, use very strict ratio
            max_val, min_val = max(values), min(values)
            ratio = max_val / (min_val + 1e-10)
            print(f"    Magnitude ratio: {ratio:.2f}")
            
            if ratio > 100.0:  # Very strict threshold
                max_client = max(magnitudes.keys(), key=lambda k: magnitudes[k])
                detections[max_client] = min(1.0, ratio / 200.0)  # Confidence based on ratio
                print(f"    🚨 EXTREME MAGNITUDE: {max_client} (ratio: {ratio:.1f})")
        
        elif len(values) >= 3:
            # For 3+ clients, use very conservative statistical methods
            median_val = np.median(values)
            
            for client_id, magnitude in magnitudes.items():
                confidence = 0.0
                reasons = []
                
                # Extremely high absolute threshold
                if magnitude > 50000:
                    confidence += 0.9
                    reasons.append(f"extreme absolute magnitude")
                
                # Very high relative threshold
                if magnitude > median_val * 50.0:
                    confidence += 0.8
                    reasons.append(f"50x median")
                
                if confidence > 0.7:  # Only very high confidence
                    detections[client_id] = confidence
                    print(f"    🚨 EXTREME MAGNITUDE: {client_id}: {', '.join(reasons)} (conf: {confidence:.2f})")
        
        return detections
    
    def _detect_extreme_patterns(self, client_updates: Dict[str, List[np.ndarray]]) -> List[str]:
        """Detect only the most extreme parameter patterns."""
        outliers = []
        
        print("  Checking for extreme patterns...")
        
        for client_id, update_list in client_updates.items():
            is_extreme = False
            
            for i, param in enumerate(update_list):
                if not isinstance(param, np.ndarray) or param.size == 0:
                    continue
                
                param_flat = param.flatten()
                max_abs = np.max(np.abs(param_flat))
                
                # Only flag truly extreme cases
                if max_abs > 1000:  # Very high threshold
                    print(f"    🚨 EXTREME VALUES in {client_id} layer {i}: max={max_abs:.1f}")
                    is_extreme = True
                    break
                
                # Check for obviously artificial patterns
                if param.size > 100:  # Only for large parameters
                    unique_vals = len(np.unique(param_flat))
                    if unique_vals < 3:  # Almost all same values
                        print(f"    🚨 ARTIFICIAL PATTERN in {client_id} layer {i}: only {unique_vals} unique values")
                        is_extreme = True
                        break
            
            if is_extreme:
                outliers.append(client_id)
        
        return outliers
    
    def _conservative_validation(self, suspicious_clients: set, 
                                confidence_scores: Dict[str, float], 
                                num_clients: int, 
                                client_metrics: Optional[Dict[str, Dict[str, float]]]) -> set:
        """Apply very conservative validation."""
        if len(suspicious_clients) == 0:
            return suspicious_clients
        
        # Rule 1: For 2 clients, be extremely conservative
        if num_clients == 2:
            print(f"    Conservative mode: 2-client scenario")
            
            # Only keep clients with explicit poisoning indicator OR extreme confidence
            validated = set()
            for client_id in suspicious_clients:
                confidence = confidence_scores.get(client_id, 0)
                has_poisoning_flag = False
                
                if client_metrics and client_id in client_metrics:
                    has_poisoning_flag = client_metrics[client_id].get('poisoning_active', False)
                
                if has_poisoning_flag:
                    validated.add(client_id)
                    print(f"    Keeping {client_id}: explicit poisoning indicator")
                elif confidence > 0.9:  # Extremely high confidence
                    validated.add(client_id)
                    print(f"    Keeping {client_id}: extreme confidence ({confidence:.2f})")
                else:
                    print(f"    Filtering out {client_id}: insufficient confidence ({confidence:.2f})")
            
            return validated
        
        # Rule 2: For 3+ clients, keep high-confidence detections
        validated = set()
        for client_id in suspicious_clients:
            confidence = confidence_scores.get(client_id, 0)
            if confidence > 0.7:  # High confidence threshold
                validated.add(client_id)
                print(f"    Keeping {client_id}: high confidence ({confidence:.2f})")
            else:
                print(f"    Filtering out {client_id}: low confidence ({confidence:.2f})")
        
        return validated
    
    def _assess_threat_conservative(self, suspicious_clients: set, 
                                  confidence_scores: Dict[str, float], 
                                  total_clients: int) -> str:
        """Conservative threat assessment."""
        if not suspicious_clients or total_clients == 0:
            return "low"
        
        # Consider confidence in threat assessment
        max_confidence = max([confidence_scores.get(c, 0) for c in suspicious_clients])
        avg_confidence = np.mean([confidence_scores.get(c, 0) for c in suspicious_clients])
        suspicious_ratio = len(suspicious_clients) / total_clients
        
        # More conservative thresholds
        if max_confidence > 0.9 and suspicious_ratio >= 0.3:
            return "critical"
        elif max_confidence > 0.8:
            return "high"  
        elif max_confidence > 0.6 or suspicious_ratio >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _log_detection_results(self, detection_details: Dict[str, Any]):
        """Log detection results."""
        round_num = detection_details["round"]
        threat_level = detection_details["threat_level"]
        suspicious_count = len(detection_details["suspicious_clients"])
        
        if threat_level in ["high", "critical"]:
            logger.error(f"SECURITY ALERT - Round {round_num}: {threat_level.upper()} threat!")
            logger.error(f"Suspicious clients: {detection_details['suspicious_clients']}")
        elif suspicious_count > 0:
            logger.warning(f"Round {round_num}: {suspicious_count} suspicious clients detected")
        else:
            logger.info(f"Round {round_num}: No threats detected")
    
    def _save_detection_log(self, detection_details: Dict[str, Any]):
        """Save detection results to file."""
        try:
            round_num = detection_details["round"]
            timestamp = datetime.now().isoformat()
            
            log_entry = {
                "timestamp": timestamp,
                **detection_details
            }
            
            log_file = os.path.join(self.log_dir, f"detection_round_{round_num}.json")
            with open(log_file, 'w') as f:
                json.dump(log_entry, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save detection log: {e}")
    
    def get_trusted_clients(self, threshold: Optional[float] = None) -> List[str]:
        """Get list of currently trusted clients."""
        if threshold is None:
            threshold = self.reputation_threshold
        return [
            client_id for client_id, reputation in self.client_reputations.items()
            if reputation >= threshold
        ]
    
    def get_client_reputation(self, client_id: str) -> float:
        """Get reputation score for a specific client."""
        return self.client_reputations.get(client_id, 1.0)
    
    def get_reputation_summary(self) -> pd.DataFrame:
        """Get summary of all client reputations."""
        data = []
        for client_id, reputation in self.client_reputations.items():
            data.append({
                'client_id': client_id,
                'reputation': reputation,
                'trusted': reputation >= self.reputation_threshold,
                'status': 'Trusted' if reputation >= self.reputation_threshold else 'Suspicious'
            })
        return pd.DataFrame(data)

# Convenience function
def create_detector() -> PoisoningDetector:
    """Create a conservative detector."""
    return PoisoningDetector()