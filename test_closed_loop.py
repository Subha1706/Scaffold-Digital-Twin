import torch
import torch.nn as nn
from closed_loop_twin import ClosedLoopDigitalTwin

# ==========================================
# 1. SETUP MOCK/TEST MODELS FOR BENCHMARKING
# ==========================================
# (Replace these with your actual loaded PyTorch model checkpoints if available)

class MockTransformer(nn.Module):
    def forward(self, x):
        # Checks if token indicates a known supervised defect (e.g., high feature variance)
        if x.mean().item() > 0.5:
            return torch.tensor([[0.88]])  # High probability of defect
        return torch.tensor([[0.12]])      # Nominal probability

class MockAutoencoder(nn.Module):
    def forward(self, x):
        # Simulates reconstruction. For unmodeled anomalies, noise causes higher MSE drift.
        if torch.abs(x).max().item() > 2.0:
            return x + 0.5  # High reconstruction error (unseen defect)
        return x + 0.02     # Clean reconstruction (nominal/known)

# ==========================================
# 2. RUN THE THREE TEST SCENARIOS
# ==========================================

def run_system_verification():
    # Initialize Controller with your calibrated 0.170449 threshold
    twin = ClosedLoopDigitalTwin(
        transformer_model=MockTransformer(),
        autoencoder_model=MockAutoencoder(),
        mse_threshold=0.170449
    )

    # Define 3 distinct physical layer scenarios (32-dim fused tokens)
    test_cases = {
        "Scenario 1: Nominal Layer (Healthy Print)": torch.zeros(1, 32),
        "Scenario 2: Known Defect (Supervised Flag)": torch.ones(1, 32) * 0.8,
        "Scenario 3: Unseen Anomaly (Out-of-Distribution Spike)": torch.randn(1, 32) * 2.5
    }

    print("\n" + "="*65)
    print("      PHASE 6 CLOSED-LOOP INTEGRATION VERIFICATION")
    print("="*65)

    for scenario_name, token in test_cases.items():
        prob, mse = twin.evaluate_layer(token)
        output = twin.compute_corrective_action(prob, mse)

        print(f"\n▶ {scenario_name}")
        print(f"  ├─ Transformer Prob (Phase 4) : {output['anomaly_probability']:.4f} (Threshold: > 0.75)")
        print(f"  ├─ Autoencoder MSE  (Phase 5) : {output['reconstruction_mse']:.6f} (Threshold: > 0.170449)")
        print(f"  ├─ System Status             : {output['status']}")
        print(f"  ├─ Target Nozzle Temp        : {output['target_nozzle_temp']} °C")
        print(f"  ├─ Feed Rate Override        : {int(output['feed_rate_override'] * 100)}%")
        print(f"  └─ Emitted G-Code            : [{output['emitted_gcode']}]")

    print("\n" + "="*65 + "\n")

if __name__ == "__main__":
    run_system_verification()