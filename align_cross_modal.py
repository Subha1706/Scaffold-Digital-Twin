# align_cross_modal.py
import os
import torch
import pandas as pd
import numpy as np
from generate_dic_maps import run_synthetic_dic_pipeline

def construct_cross_modal_dataset():
    print("====================================================")
    print("      RE-ALIGNING CROSS-MODAL DATASET (STEP 4)     ")
    print("====================================================")
    
    tabular_path = "base_print_parameters.csv"
    if not os.path.exists(tabular_path):
        raise FileNotFoundError("Run 'python process_tabular.py' first.")
    
    param_df = pd.read_csv(tabular_path)
    num_samples = len(param_df)
    
    feature_cols = [
        'nozzle_temperature_c', 'bed_temperature_c', 'print_speed_mm_s', 
        'infill_time_sec', 'travel_time_sec', 'dimensional_deviation_mm'
    ]
    tabular_features = param_df[feature_cols].values
    
    peak_strain_scalar = run_synthetic_dic_pipeline()
    
    aligned_tabular = []
    aligned_mechanical = []
    aligned_labels = []
    
    np.random.seed(42)
    for idx in range(num_samples):
        row_features = tabular_features[idx].copy()
        
        # Inject an aggressive 50/50 split of anomalies to force the transformer to learn
        is_defect = 1 if (idx % 2 == 0) else 0
        
        if is_defect:
            row_features[2] += 15.0  # Spike print speed
            row_features[5] += 0.05  # Spike dimensional deviation
            sample_strain = peak_strain_scalar * 2.5 + np.random.normal(0, 0.1)
        else:
            row_features[2] -= 5.0
            sample_strain = peak_strain_scalar * 0.5 + np.random.normal(0, 0.1)
        
        aligned_tabular.append(row_features)
        aligned_mechanical.append([sample_strain])
        aligned_labels.append(is_defect)
        
    # Apply manual Z-score standardization so features share a common variance scale
    X_tab_np = np.array(aligned_tabular)
    X_tab_mean = X_tab_np.mean(axis=0)
    X_tab_std = X_tab_np.std(axis=0) + 1e-6
    X_tab_scaled = (X_tab_np - X_tab_mean) / X_tab_std
    
    X_mech_np = np.array(aligned_mechanical)
    X_mech_mean = X_mech_np.mean(axis=0)
    X_mech_std = X_mech_np.std(axis=0) + 1e-6
    X_mech_scaled = (X_mech_np - X_mech_mean) / X_mech_std
    
    torch.save({
        'tabular_features': torch.tensor(X_tab_scaled, dtype=torch.float32),
        'mechanical_features': torch.tensor(X_mech_scaled, dtype=torch.float32),
        'labels': torch.tensor(aligned_labels, dtype=torch.long),
        'tab_mean': torch.tensor(X_tab_mean, dtype=torch.float32),
        'tab_std': torch.tensor(X_tab_std, dtype=torch.float32),
        'mech_mean': torch.tensor(X_mech_mean, dtype=torch.float32),
        'mech_std': torch.tensor(X_mech_std, dtype=torch.float32)
    }, "scaffold_twin_dataset.pt")
    
    print("[✔] SUCCESS: Dataset updated with explicit contrast and scaling factors.")
    print("====================================================")

if __name__ == "__main__":
    construct_cross_modal_dataset()