# train_phase4_extensions.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# =======================================================
# PHASE 4: UNSUPERVISED CROSS-MODAL AUTOENCODER MODEL
# =======================================================
class DigitalTwinAutoencoder(nn.Module):
    """
    Compresses cross-modal telemetry profiles into a tight latent space
    and reconstructs them to isolate anomalies via mathematical reconstruction errors.
    """
    def __init__(self, input_dim=7, latent_dim=3): # 6 tabular features + 1 µDIC strain feature = 7 total dims
        super(DigitalTwinAutoencoder, self).__init__()
        
        # Encoder Network: Compress features into bottleneck state
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        
        # Decoder Network: Reconstruct original continuous input values
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
        
    def forward(self, x):
        latent_bottleneck = self.encoder(x)
        reconstructed_vector = self.decoder(latent_bottleneck)
        return reconstructed_vector

def execute_phase4_pipeline():
    print("====================================================")
    print("     EXECUTING EXTENDED ANOMALY PIPELINE (PHASE 4)   ")
    print("====================================================")
    
    dataset_path = "scaffold_twin_dataset.pt"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Missing baseline dataset. Please run 'python align_cross_modal.py' first.")
        
    data_pack = torch.load(dataset_path)
    X_tab = data_pack['tabular_features']
    X_mech = data_pack['mechanical_features']
    labels = data_pack['labels']
    
    # 1. SYNTHETIC TRAJECTORY AUGMENTATION (Solving small-data variance constraints)
    print("[+] Initiating Synthetic Trajectory Augmentation Engine...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Combine tabular and mechanical modalities into a single cross-modal profile matrix
    raw_combined = torch.cat((X_tab, X_mech), dim=1)
    
    # Isolate nominal data profiles to teach the autoencoder what a "perfect print" looks like
    nominal_indices = (labels == 0).nonzero(as_tuple=True)[0]
    nominal_base_data = raw_combined[nominal_indices]
    
    # Programmatically expand data footprints by adding controlled Gaussian variations
    augmented_nominal_list = []
    for _ in range(20): # Generate 20 variants for each base record (~500 augmented clean samples)
        noise = torch.randn_like(nominal_base_data) * 0.05
        augmented_nominal_list.append(nominal_base_data + noise)
        
    X_train_unsupervised = torch.cat(augmented_nominal_list, dim=0)
    print(f"    -> Expanded training dataset profile to {X_train_unsupervised.shape[0]} robust nominal samples.")
    
    # 2. OPTIMIZING THE AUTOENCODER ARCHITECTURE
    train_loader = DataLoader(TensorDataset(X_train_unsupervised), batch_size=16, shuffle=True)
    model = DigitalTwinAutoencoder(input_dim=7, latent_dim=3)
    
    criterion = nn.MSELoss() # Mean Squared Error calculates reconstruction differences
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    
    print("[+] Training unsupervised network to recognize nominal structural states...")
    model.train()
    for epoch in range(1, 31):
        epoch_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0]
            optimizer.zero_grad()
            reconstructed = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
            
    print(f"    -> Optimization finished. Convergence achieved at Baseline MSE Loss: {epoch_loss / len(X_train_unsupervised):.5f}")
    
    # 3. STATIC THRESHOLD CALIBRATION
    # Calculate reconstruction boundaries across normal data to find the cutoff line
    model.eval()
    with torch.no_grad():
        nominal_reconstructed = model(nominal_base_data)
        errors = torch.mean((nominal_base_data - nominal_reconstructed) ** 2, dim=1).numpy()
        
    # Set the anomaly threshold at the 95th percentile of normal reconstruction variations
    anomaly_threshold = float(np.percentile(errors, 95))
    print(f"[+] Statistical Anomaly Threshold Calibrated: {anomaly_threshold:.6f}")
    
    # 4. EXPORTING PHASE 4 FRAMEWORK WEIGHTS
    output_meta_path = "scaffold_twin_autoencoder.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'anomaly_threshold': anomaly_threshold,
        'tab_mean': data_pack['tab_mean'],
        'tab_std': data_pack['tab_std'],
        'mech_mean': data_pack['mech_mean'],
        'mech_std': data_pack['mech_std']
    }, output_meta_path)
    
    print("----------------------------------------------------")
    print(f"[✔] SUCCESS: Unsupervised Autoencoder Framework Operational!")
    print(f"    Saved Asset Pack Path: {os.path.abspath(output_meta_path)}")
    print("====================================================")

if __name__ == "__main__":
    execute_phase4_pipeline()