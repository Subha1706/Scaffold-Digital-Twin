import numpy as np
import torch
import torch.nn as nn

# ==========================================
# 1. MOCK ENVIRONMENT & MODEL INTERACTION
# ==========================================

class ClosedLoopDigitalTwin:
    def __init__(self, transformer_model, autoencoder_model, mse_threshold=0.170449):
        self.transformer = transformer_model
        self.autoencoder = autoencoder_model
        self.threshold = mse_threshold
        
        # Nominal operational parameters
        self.nominal_temp = 210.0  # Celsius
        self.nominal_feed_rate = 1.0  # 100% velocity multiplier

    def evaluate_layer(self, fused_token):
        """
        Runs Phase 4 Transformer & Phase 5 Autoencoder inference 
        on the 32-dimensional fused token.
        """
        with torch.no_grad():
            # Phase 4: Diagnostic probability (Supervised)
            anomaly_prob = self.transformer(fused_token).item()
            
            # Phase 5: Reconstruction Error (Unsupervised)
            reconstructed_token = self.autoencoder(fused_token)
            rec_error = torch.mean((fused_token - reconstructed_token) ** 2).item()
            
        return anomaly_prob, rec_error

    def compute_corrective_action(self, anomaly_prob, rec_error):
        """
        Phase 6 Actuator: Intercepts probabilities/errors 
        and calculates dynamic G-Code parameters.
        """
        # Flag conditions
        supervised_flag = anomaly_prob > 0.75
        unsupervised_flag = rec_error > self.threshold

        if supervised_flag or unsupervised_flag:
            # Calculate dynamic corrections based on risk level
            risk_severity = max(anomaly_prob, min(rec_error / (self.threshold * 2), 1.0))
            
            # Action: Boost thermal window & throttle feed rate proportionally
            corrected_temp = self.nominal_temp + (risk_severity * 5.0)  # Up to +5°C
            corrected_feed = max(0.70, self.nominal_feed_rate - (risk_severity * 0.25))  # Down to 70%
            
            gcode_command = f"M104 S{corrected_temp:.1f} ; M220 S{int(corrected_feed * 100)}"
            status = "CORRECTIVE_OVERRIDE_TRIGGERED"
        else:
            corrected_temp = self.nominal_temp
            corrected_feed = self.nominal_feed_rate
            gcode_command = f"M104 S{corrected_temp:.1f} ; M220 S100"
            status = "NOMINAL_EXECUTION"

        return {
            "status": status,
            "anomaly_probability": round(anomaly_prob, 4),
            "reconstruction_mse": round(rec_error, 6),
            "target_nozzle_temp": corrected_temp,
            "feed_rate_override": corrected_feed,
            "emitted_gcode": gcode_command
        }

# ==========================================
# 2. SIMULATION RUNNER
# ==========================================

if __name__ == "__main__":
    # Dummy placeholder models to demonstrate execution logic
    class DummyTransformer(nn.Module):
        def forward(self, x): return torch.tensor([[0.82]]) # Simulating defect probability

    class DummyAutoencoder(nn.Module):
        def forward(self, x): return x + 0.15 # Simulating reconstruction drift

    # Initialize twin with pre-trained instances
    twin_controller = ClosedLoopDigitalTwin(
        transformer_model=DummyTransformer(),
        autoencoder_model=DummyAutoencoder(),
        mse_threshold=0.170449
    )

    # Simulated 32-dimensional fused token for Layer 14
    sample_fused_token = torch.randn(1, 32)

    # Execute Closed-Loop Diagnostic
    prob, mse = twin_controller.evaluate_layer(sample_fused_token)
    control_output = twin_controller.compute_corrective_action(prob, mse)

    # Display Diagnostic Log
    print("=== PHASE 6: CLOSED-LOOP DIAGNOSTIC LOG ===")
    for key, value in control_output.items():
        print(f"{key:<22}: {value}")