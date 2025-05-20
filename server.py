import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import logging
import os
import sys
import json
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from collections import OrderedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FraminghamServer")

# Server configuration
SERVER_CONFIG = {
    "num_rounds": 10,
    "min_available_clients": 2,
    "min_fit_clients": 2,
    "min_evaluate_clients": 2,
    "input_size": 15,  # This should match the number of features in your dataset
    "save_dir": "model_checkpoints",
    "aggregation": "fedprox",  # Options: "fedavg", "fedprox"
    "proximal_mu": 0.01  # FedProx hyperparameter (should match clients)
}

# Same model definition as the client
class HeartDiseaseModel(nn.Module):
    def __init__(self, input_size):
        super(HeartDiseaseModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Aggregation function for FedProx (server-side)
def federated_averaging(results):
    """Compute weighted average of model updates based on number of examples."""
    # Extract weights and num_examples
    weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) 
                      for _, fit_res in results]
    
    # Get the total number of examples
    total_examples = sum([num_examples for _, num_examples in weights_results])
    
    # Log the weights distribution
    logger.info(f"Aggregating from {len(weights_results)} clients with a total of {total_examples} examples")
    
    if total_examples == 0:
        # Handle edge case of no examples
        return None
    
    # Compute weighted average
    weighted_weights = []
    for weights, num_examples in weights_results:
        # Skip clients that had no examples
        if num_examples == 0:
            continue
            
        # Calculate weight for this client based on number of examples
        weight = num_examples / total_examples
        
        # Apply weighted contribution to each layer
        if not weighted_weights:  # First client, initialize weighted_weights
            weighted_weights = [layer * weight for layer in weights]
        else:  # Add to the existing weighted sum
            for i, layer in enumerate(weights):
                weighted_weights[i] += layer * weight
    
    return weighted_weights

# Custom strategy for FedProx
class FedProxStrategy(fl.server.strategy.FedAvg):
    """Custom strategy for Framingham study with FedProx support"""
    
    def __init__(self, initial_parameters, save_dir=None):
        super().__init__(
            fraction_fit=1.0,  # Use all available clients for training
            fraction_evaluate=1.0,  # Use all available clients for evaluation
            min_fit_clients=SERVER_CONFIG["min_fit_clients"],
            min_evaluate_clients=SERVER_CONFIG["min_evaluate_clients"],
            min_available_clients=SERVER_CONFIG["min_available_clients"],
        )
        self.initial_parameters = initial_parameters
        self.save_dir = save_dir or "model_checkpoints"
        self.metrics_history = []
        
        # Create directory for saving models
        os.makedirs(self.save_dir, exist_ok=True)
    
    def initialize_parameters(self, client_manager):
        """Initialize model parameters"""
        return self.initial_parameters
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model updates from clients"""
        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate!")
            return None, {}
        
        # Log aggregation info
        logger.info(f"Round {server_round}: Aggregating updates from {len(results)} clients")
        
        # Extract client metrics for logging
        client_metrics = {}
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            metrics = fit_res.metrics if fit_res.metrics else {}
            client_metrics[client_id] = metrics
            
            # Log details for each client
            logger.info(f"Client {client_id} - Examples: {fit_res.num_examples}")
            for key, value in metrics.items():
                logger.info(f"Client {client_id} - {key}: {value}")
        
        # Perform aggregation - same as FedAvg for the server side
        # (FedProx's main difference is in the client's loss function)
        if len(results) > 0:
            # Aggregate parameters using federated averaging
            aggregated_ndarrays = federated_averaging(results)
            
            if aggregated_ndarrays is not None:
                aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
                
                # Add round number to metrics
                metrics = {"round": server_round}
                
                # Calculate average metrics across all clients 
                if len(client_metrics) > 0:
                    avg_accuracy = np.mean([
                        m.get("accuracy", 0.0) for m in client_metrics.values()
                    ])
                    avg_loss = np.mean([
                        m.get("loss", 0.0) for m in client_metrics.values()
                    ])
                    metrics["accuracy"] = float(avg_accuracy)
                    metrics["loss"] = float(avg_loss)
                
                # Save the aggregated model
                self._save_model(server_round, aggregated_ndarrays)
                
                # Store metrics for history
                self.metrics_history.append({
                    "round": server_round,
                    "client_metrics": client_metrics,
                    "aggregated_metrics": metrics,
                })
                self._save_metrics()
                
                return aggregated_parameters, metrics
        
        # Fallback if aggregation fails
        return None, {}
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results from clients"""
        if not results:
            return None, {}
        
        # Log evaluation info
        logger.info(f"Round {server_round}: Aggregating evaluation from {len(results)} clients")
        
        # Extract and aggregate metrics across clients
        accuracies = [r.metrics.get("accuracy", 0.0) for _, r in results]
        losses = [r.loss for _, r in results]
        
        # Compute averages
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        # Final metrics
        metrics = {
            "accuracy": float(avg_accuracy),
            "loss": float(avg_loss),
        }
        
        logger.info(f"Round {server_round} evaluation - Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}")
        
        return avg_loss, metrics
    
    def _save_model(self, round_num, parameters):
        """Save model checkpoint"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.save_dir, exist_ok=True)
            
            # Convert parameters to PyTorch model
            model = HeartDiseaseModel(input_size=SERVER_CONFIG["input_size"])
            
            # Fix the parameter shapes before saving
            # This ensures parameters have consistent format and avoids inhomogeneous array errors
            state_dict = OrderedDict()
            model_state_dict = model.state_dict()
            
            for i, (name, _) in enumerate(model_state_dict.items()):
                if i < len(parameters):
                    state_dict[name] = torch.tensor(parameters[i])
            
            model.load_state_dict(state_dict)
            
            # Save PyTorch model
            model_path = os.path.join(self.save_dir, f"model_round_{round_num}.pt")
            torch.save(model.state_dict(), model_path)
            
            logger.info(f"Saved model checkpoint for round {round_num}")
            
            # Also save raw parameters as NumPy arrays 
            params_path = os.path.join(self.save_dir, f"params_round_{round_num}.npz")
            np.savez(params_path, *parameters)
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}", exc_info=True)
    
    def _save_metrics(self):
        """Save training metrics history"""
        try:
            metrics_path = os.path.join(self.save_dir, "metrics_history.json")
            with open(metrics_path, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.info(f"Saved metrics history to {metrics_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")

def start_server(
    num_rounds=None, 
    min_clients=None, 
    input_size=None, 
    save_dir=None,
    aggregation=None,
    proximal_mu=None
):
    """Start the federated learning server"""
    # Update configuration if values provided
    if num_rounds is not None:
        SERVER_CONFIG["num_rounds"] = num_rounds
    if min_clients is not None:
        SERVER_CONFIG["min_available_clients"] = min_clients
        SERVER_CONFIG["min_fit_clients"] = min_clients
        SERVER_CONFIG["min_evaluate_clients"] = min_clients
    if input_size is not None:
        SERVER_CONFIG["input_size"] = input_size
    if save_dir is not None:
        SERVER_CONFIG["save_dir"] = save_dir
    if aggregation is not None:
        SERVER_CONFIG["aggregation"] = aggregation
    if proximal_mu is not None:
        SERVER_CONFIG["proximal_mu"] = proximal_mu
    
    # Create directory for saving models
    os.makedirs(SERVER_CONFIG["save_dir"], exist_ok=True)
    
    # Initialize model 
    model = HeartDiseaseModel(input_size=SERVER_CONFIG["input_size"])
    
    # Get model parameters as numpy arrays
    initial_params = [val.detach().cpu().numpy() for val in model.parameters()]
    
    # Convert to Flower parameters format
    initial_parameters = ndarrays_to_parameters(initial_params)
    
    # Create strategy (using the same strategy regardless of FedAvg or FedProx since the
    # main FedProx logic is in the client loss function)
    strategy = FedProxStrategy(
        initial_parameters=initial_parameters,
        save_dir=SERVER_CONFIG["save_dir"]
    )
    
    # Print configuration
    print(f"\n===== Framingham Heart Study FL Server ({SERVER_CONFIG['aggregation']}) =====")
    print(f"Rounds:          {SERVER_CONFIG['num_rounds']}")
    print(f"Minimum clients: {SERVER_CONFIG['min_available_clients']}")
    print(f"Input size:      {SERVER_CONFIG['input_size']}")
    print(f"Save directory:  {SERVER_CONFIG['save_dir']}")
    if SERVER_CONFIG['aggregation'] == 'fedprox':
        print(f"Proximal mu:     {SERVER_CONFIG['proximal_mu']}")
    print("=============================================")
    print("\nWaiting for clients to connect...\n")
    
    # Start server
    server_config = fl.server.ServerConfig(num_rounds=SERVER_CONFIG["num_rounds"])
    return fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=server_config
    )

if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Framingham Heart Study FL Server")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated learning rounds")
    parser.add_argument("--min_clients", type=int, default=2, help="Minimum number of clients")
    parser.add_argument("--input_size", type=int, default=15, help="Number of input features")
    parser.add_argument("--save_dir", type=str, default="model_checkpoints", help="Directory to save models")
    parser.add_argument("--aggregation", type=str, default="fedprox", choices=["fedavg", "fedprox"], 
                      help="Aggregation method")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal term strength")
    
    # Handle case when running in Jupyter
    if 'ipykernel' in sys.modules:
        print("Running in Jupyter/IPython environment")
        start_server()
    else:
        # Parse arguments and start server
        args = parser.parse_args()
        
        try:
            start_server(
                num_rounds=args.rounds,
                min_clients=args.min_clients, 
                input_size=args.input_size,
                save_dir=args.save_dir,
                aggregation=args.aggregation,
                proximal_mu=args.mu
            )
        except Exception as e:
            logger.error(f"Server failed: {str(e)}", exc_info=True)