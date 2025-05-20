import torch
import torch.nn as nn
import flwr as fl
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FraminghamClient")

# Client configuration
CLIENT_CONFIG = {
    "local_epochs": 3,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "dropout_rate": 0.3,
    "server_address": "localhost:8080",
    "proximal_mu": 0.01  # FedProx hyperparameter (controls the proximal term strength)
}

# Model for Framingham Heart Study data
class HeartDiseaseModel(nn.Module):
    def __init__(self, input_size):
        super(HeartDiseaseModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(CLIENT_CONFIG["dropout_rate"]),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(CLIENT_CONFIG["dropout_rate"]),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# FedProx Loss Function
class FedProxLoss(nn.Module):
    def __init__(self, base_criterion, mu=0.01):
        super(FedProxLoss, self).__init__()
        self.base_criterion = base_criterion
        self.mu = mu  # Proximal term coefficient
        
    def forward(self, y_pred, y_true, model_params, global_params):
        # Calculate the base loss (e.g., BCE loss)
        base_loss = self.base_criterion(y_pred, y_true)
        
        # Calculate the proximal term if global parameters are provided
        proximal_term = 0.0
        if global_params is not None:
            # Sum up the squared L2 norm of the difference between local and global model parameters
            for local_param, global_param in zip(model_params, global_params):
                proximal_term += torch.sum((local_param - global_param) ** 2)
                
            # Add the weighted proximal term to the base loss
            loss = base_loss + (self.mu / 2) * proximal_term
            return loss
        
        # If no global parameters are provided, just return the base loss
        return base_loss

# Load and preprocess Framingham data
def load_data(data_path):
    """Load and preprocess Framingham Heart Study data"""
    try:
        # Read CSV data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {data_path} with shape {df.shape}")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            logger.info(f"Found {missing_values} missing values, dropping rows with missing values")
            df.dropna(inplace=True)
            logger.info(f"Shape after dropping missing values: {df.shape}")
        
        # Ensure the target column exists
        if "TenYearCHD" not in df.columns:
            raise ValueError("Target column 'TenYearCHD' not found in dataset!")
            
        # Split features and target
        X = df.drop(columns=["TenYearCHD"])
        y = df["TenYearCHD"]
        
        # Show class distribution
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=CLIENT_CONFIG["batch_size"], shuffle=True)
        
        logger.info(f"Created dataloader with {len(dataset)} samples and {X.shape[1]} features")
        return dataloader, X.shape[1]
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# Client class for Federated Learning with FedProx
class FraminghamClient(fl.client.NumPyClient):
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.global_params = None  # Store global model parameters for FedProx
        logger.info(f"Initialized client with device: {device}")
        
    def get_parameters(self, config):
        """Get model parameters as a list of NumPy arrays"""
        # Using detach() to prevent gradient error
        return [val.detach().cpu().numpy() for val in self.model.parameters()]
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays"""
        # Convert to torch tensors
        self.global_params = [torch.tensor(p, device=self.device) for p in parameters]
        
        # Update model
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        logger.info("Parameters updated from server")
        
    def fit(self, parameters, config):
        """Train the model on local data with FedProx"""
        # Update model with server parameters
        self.set_parameters(parameters)
        
        # Train the model
        self.model.train()
        
        # Standard loss function
        criterion = nn.BCELoss()
        
        # FedProx loss function
        proximal_criterion = FedProxLoss(criterion, mu=CLIENT_CONFIG["proximal_mu"])
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=CLIENT_CONFIG["learning_rate"],
            weight_decay=CLIENT_CONFIG["weight_decay"]
        )
        
        # Metrics for tracking
        total_loss = 0.0
        total_samples = 0
        correct = 0
        
        # Train for multiple epochs
        for epoch in range(CLIENT_CONFIG["local_epochs"]):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (X, y) in enumerate(self.dataloader):
                # Move tensors to device
                X, y = X.to(self.device), y.to(self.device)
                
                # Forward pass
                y_pred = self.model(X)
                
                # Calculate loss with proximal term
                loss = proximal_criterion(
                    y_pred, 
                    y, 
                    self.model.parameters(),  # Current model parameters
                    self.global_params        # Global model parameters
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                batch_loss = loss.item() * X.size(0)
                total_loss += batch_loss
                epoch_loss += batch_loss
                total_samples += X.size(0)
                epoch_samples += X.size(0)
                
                # Calculate accuracy
                predicted = (y_pred > 0.5).float()
                correct += (predicted == y).sum().item()
                
                # Log progress occasionally
                if batch_idx % 5 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{CLIENT_CONFIG['local_epochs']} - "
                        f"Batch {batch_idx}/{len(self.dataloader)} - "
                        f"Loss: {loss.item():.4f}"
                    )
            
            # Log epoch metrics
            logger.info(
                f"Epoch {epoch+1}/{CLIENT_CONFIG['local_epochs']} completed - "
                f"Loss: {epoch_loss/epoch_samples:.4f}"
            )
        
        # Calculate final metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = correct / total_samples if total_samples > 0 else 0
        
        logger.info(f"Training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Return updated model parameters and metrics
        return self.get_parameters({}), total_samples, {"loss": float(avg_loss), "accuracy": float(accuracy)}
    
    def evaluate(self, parameters, config):
        """Evaluate the model on local data"""
        # Update model with server parameters
        self.set_parameters(parameters)
        
        # Evaluate the model
        self.model.eval()
        criterion = nn.BCELoss()
        
        loss = 0.0
        total = 0
        correct = 0
        
        with torch.no_grad():
            for X, y in self.dataloader:
                # Move tensors to device
                X, y = X.to(self.device), y.to(self.device)
                
                # Forward pass
                y_pred = self.model(X)
                batch_loss = criterion(y_pred, y).item()
                
                # Update metrics
                loss += batch_loss * X.size(0)
                total += X.size(0)
                
                # Calculate accuracy
                predicted = (y_pred > 0.5).float()
                correct += (predicted == y).sum().item()
        
        # Calculate final metrics
        avg_loss = loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Return metrics
        return float(avg_loss), total, {"accuracy": float(accuracy)}

def start_client(client_id=0, server_address=None):
    """Initialize and start a client"""
    # Update server address if provided
    if server_address:
        CLIENT_CONFIG["server_address"] = server_address
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Determine which data file to use based on client ID
    data_path = f"framingham_part{client_id+1}.csv"
    
    # Try alternative naming if file doesn't exist
    if not os.path.exists(data_path):
        alternative_path = f"framingham_part{client_id+1}.csv"
        if os.path.exists(alternative_path):
            data_path = alternative_path
        else:
            logger.error(f"Data file {data_path} not found")
            return
    
    # Load data
    dataloader, input_size = load_data(data_path)
    
    # Initialize model
    model = HeartDiseaseModel(input_size=input_size).to(device)
    logger.info(f"Model initialized with input size: {input_size}")
    
    # Create client
    client = FraminghamClient(model, dataloader, device)
    
    # Start client
    logger.info(f"Starting client {client_id} and connecting to {CLIENT_CONFIG['server_address']}")
    
    print(f"\n===== Framingham Heart Study FL Client {client_id} (FedProx) =====")
    print(f"Server:        {CLIENT_CONFIG['server_address']}")
    print(f"Data file:     {data_path}")
    print(f"Local epochs:  {CLIENT_CONFIG['local_epochs']}")
    print(f"Batch size:    {CLIENT_CONFIG['batch_size']}")
    print(f"Proximal mu:   {CLIENT_CONFIG['proximal_mu']}")
    print(f"Device:        {device}")
    print("=================================================")
    print(f"\nConnecting to server...\n")
    
    fl.client.start_client(server_address=CLIENT_CONFIG["server_address"], client=client)

# For Jupyter usage
if __name__ == "__main__":
    # Check if running in Jupyter
    if 'ipykernel' in sys.modules:
        print("Running in Jupyter/IPython environment")
        # Default to client ID 0, can be changed by user
        start_client(client_id=0)
    else:
        # For command line use
        import argparse
        parser = argparse.ArgumentParser(description="Framingham Heart Study FL Client")
        parser.add_argument("--id", type=int, default=0, help="Client ID (0, 1, or 2)")
        parser.add_argument("--server", type=str, default="localhost:8080", help="Server address")
        parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal term strength")
        
        args = parser.parse_args()
        
        if args.id not in [0, 1, 2]:
            logger.error("Client ID must be 0, 1, or 2")
        else:
            try:
                # Set FedProx hyperparameter
                CLIENT_CONFIG["proximal_mu"] = args.mu
                
                start_client(args.id, args.server)
            except Exception as e:
                logger.error(f"Client failed: {str(e)}")