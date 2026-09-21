# simulate_production_twin.py
import torch
import numpy as np
import time
import os
from train_transformer import ScaffoldDigitalTwinTransformer
from train_phase4_extensions import DigitalTwinAutoencoder

def run_large_scale_twin_simulation():
    print("====================================================")
    print("    PRODUCTION-SCALE DIGITAL TWIN INFERENCE (PHASE 5)")
    print("====================================================")
    
    # 1. Load trained models and metadata scale vectors
    if not os.path.exists("scaffold_twin_dataset.pt") or not os.path.exists("scaffold_twin_autoencoder.pt"):
        raise FileNotFoundError("Missing trained model assets. Run previous phases first.")
        
    meta = torch.load("scaffold_twin_dataset.pt")
    
    transformer = ScaffoldDigitalTwinTransformer()
    transformer.load_state_dict(torch.load("scaffold_twin_transformer.pth"))
    transformer.eval()
    
    ae_pack = torch.load("scaffold_twin_autoencoder.pt")
    autoencoder = DigitalTwinAutoencoder(input_dim=7, latent_dim=3)
    autoencoder.load_state_dict(ae_pack['model_state_dict'])
    autoencoder.eval()
    ae_threshold = ae_pack['anomaly_threshold']
    
    # 2. Simulate 5,000 continuous high-frequency streaming sensor steps
    sim_steps = 5000
    print(f"[+] Simulating continuous streaming run of {sim_steps} manufacturing steps...")
    
    np.random.seed(101)
    
    # Generate baseline nominal arrays
    base_tab = np.array([210.0, 60.0, 50.0, 240.0, 15.0, 0.03])
    base_mech = np.array([1.8])
    
    latencies = []
    anomalies_caught = 0
    false_alarms = 0
    
    print("[+] Processing live synchronization pipeline loops...")
    
    with torch.no_grad():
        for step in range(sim_steps):
            start_time = time.perf_counter()
            
            # Dynamically induce a massive thermal and mechanical defect midway through production
            if step >= 2500 and step <= 2700:
                # Defect State: Temperature drops, speed spikes, strain warps aggressively
                current_tab = base_tab + np.array([-8.0, 0.2, 15.0, -40.0, -3.0, 0.09]) + np.random.normal(0, 0.02, 6)
                current_mech = base_mech + np.array([5.5]) + np.random.normal(0, 0.05, 1)
                is_true_anomaly = True
            else:
                # Nominal State: Standard operational noise
                current_tab = base_tab + np.random.normal(0, 0.05, 6)
                current_mech = base_mech + np.random.normal(0, 0.02, 1)
                is_true_anomaly = False
                
            # Cast arrays to PyTorch evaluation tensors
            x_tab_raw = torch.tensor([current_tab], dtype=torch.float32)
            x_mech_raw = torch.tensor([current_mech], dtype=torch.float32)
            
            # Apply training Z-score scale standardizations
            norm_tab = (x_tab_raw - meta['tab_mean']) / meta['tab_std']
            norm_mech = (x_mech_raw - meta['mech_mean']) / meta['mech_std']
            
            # Execute Model Engine evaluations
            combined = torch.cat((norm_tab, norm_mech), dim=1)
            rec = autoencoder(combined)
            mse_error = float(torch.mean((combined - rec) ** 2, dim=1).item())
            
            logits = transformer(norm_tab, norm_mech)
            prob_anomaly = torch.softmax(logits, dim=1).numpy()[0][1]
            
            # Anomaly trigger logic: flagged if either model senses a structural issue
            fused_alarm = (mse_error > ae_threshold) or (prob_anomaly > 0.85)
            
            if fused_alarm:
                if is_true_anomaly:
                    anomalies_caught += 1
                else:
                    false_alarms += 1
                    
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000.0)
            
    print("\n----------------------------------------------------")
    print("      PHASE 5 HIGH-VOLUME RUNTIME STATISTICS        ")
    print("----------------------------------------------------")
    print(f"  -> Total Streamed Samples Processed : {sim_steps}")
    print(f"  -> Simulated True Anomalies Induced : 201")
    print(f"  -> Anomalies Successfully Identified: {anomalies_caught}")
    print(f"  -> False Positive Alarms Triggered  : {false_alarms}")
    print(f"  -> Mean Processing Latency per Step : {sum(latencies)/len(latencies):.4f} ms")
    print("====================================================")

if __name__ == "__main__":
    run_large_scale_twin_simulation();