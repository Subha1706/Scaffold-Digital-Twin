# predict_twin.py
import torch
import numpy as np
from train_transformer import ScaffoldDigitalTwinTransformer
from train_phase4_extensions import DigitalTwinAutoencoder

def live_digital_twin_inference():
    print("====================================================")
    print("   DUAL-ENGINE DIGITAL TWIN DIAGNOSTICS (PHASE 4/5) ")
    print("====================================================")
    
    # 1. Load the core dataset metadata and normalization scales
    meta = torch.load("scaffold_twin_dataset.pt")
    
    # 2. Instantiate and load the Supervised Transformer Block
    transformer = ScaffoldDigitalTwinTransformer()
    transformer.load_state_dict(torch.load("scaffold_twin_transformer.pth"))
    transformer.eval()
    
    # 3. Instantiate and load the Unsupervised Autoencoder Block
    ae_pack = torch.load("scaffold_twin_autoencoder.pt")
    autoencoder = DigitalTwinAutoencoder(input_dim=7, latent_dim=3)
    autoencoder.load_state_dict(ae_pack['model_state_dict'])
    autoencoder.eval()
    ae_threshold = ae_pack['anomaly_threshold']
    
    print(f"[+] Active Autoencoder Threshold Bounds: {ae_threshold:.6f}")
    print("[+] Ingesting live multi-modal telemetry test parameters...\n")
    
    # Raw real-world streaming input channels to test generalization
    # Telemetry A: Completely stable nominal layer parameters
    # Telemetry B: Highly anomalous parameter line (High speed, massive strain drift)
    raw_nominal_tab = torch.tensor([[210.5, 60.1, 45.0, 245.0, 18.2, 0.02]], dtype=torch.float32)
    raw_nominal_mech = torch.tensor([[1.6]], dtype=torch.float32)
    
    raw_anomaly_tab = torch.tensor([[205.1, 59.8, 65.0, 195.5, 12.1, 0.12]], dtype=torch.float32)
    raw_anomaly_mech = torch.tensor([[8.1]], dtype=torch.float32)
    
    # Apply standard training Z-score scale transformations
    norm_nom_tab = (raw_nominal_tab - meta['tab_mean']) / meta['tab_std']
    norm_nom_mech = (raw_nominal_mech - meta['mech_mean']) / meta['mech_std']
    
    norm_anom_tab = (raw_anomaly_tab - meta['tab_mean']) / meta['tab_std']
    norm_anom_mech = (raw_anomaly_mech - meta['mech_mean']) / meta['mech_std']
    
    scenarios = {
        "Telemetry Stream A (Nominal Layer Run)": (norm_nom_tab, norm_nom_mech),
        "Telemetry Stream B (Anomaly Alert Run)": (norm_anom_tab, norm_anom_mech)
    }
    
    with torch.no_grad():
        for name, (tab, mech) in scenarios.items():
            # Merge vectors cleanly for the Autoencoder evaluation step
            combined_vector = torch.cat((tab, mech), dim=1)
            
            # Engine 1 Calculation: Unsupervised Reconstruction Error Bounds
            reconstructed = autoencoder(combined_vector)
            rec_error = float(torch.mean((combined_vector - reconstructed) ** 2, dim=1).item())
            ae_verdict = "⚠️ DRIFT ALERT" if rec_error > ae_threshold else "✔ NORMAL STABILITY"
            
            # Engine 2 Calculation: Supervised Transformer Attention Logits
            logits = transformer(tab, mech)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            tf_verdict = "❌ CRITICAL DEFECT" if probs[1] > 0.5 else "✔ STABLE LAYER"
            
            print(f"[➔] Analyzing: {name}")
            print(f"    -> [Engine 1] Autoencoder MSE Error: {rec_error:.6f} | Verdict: {ae_verdict}")
            print(f"    -> [Engine 2] Transformer Confidence : Anomaly {probs[1]*100:.2f}% | Verdict: {tf_verdict}")
            print("-" * 52)
            
    print("====================================================")

if __name__ == "__main__":
    live_digital_twin_inference()