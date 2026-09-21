import matplotlib.pyplot as plt
import numpy as np

# Set styling for clear academic presentation
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

np.random.seed(42)

# ==========================================
# GRAPH 1: STREAM A (NOMINAL HENCKY STRAIN)
# ==========================================
def save_stream_a():
    data_a = np.random.uniform(0.35, 0.55, size=(10, 10))
    data_a[2, 3] = 0.62
    data_a[7, 1] = 0.30

    plt.figure(figsize=(6, 5))
    im = plt.imshow(data_a, cmap='viridis', extent=[0, 300, 0, 300], origin='lower')
    cbar = plt.colorbar(im)
    cbar.set_label('True Strain Amplitude ($\epsilon$)', fontsize=11)
    
    plt.title('Stream A: Nominal Layer Full-Field Hencky Strain ($\mu$DIC)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('X-Coordinate Spatial Grid (pixels)', fontsize=10)
    plt.ylabel('Y-Coordinate Spatial Grid (pixels)', fontsize=10)
    plt.tight_layout()
    plt.savefig('graph1_stream_a_nominal.jpg', dpi=300, format='jpg')
    print(" Saved: graph1_stream_a_nominal.jpg")
    plt.close()

# ==========================================
# GRAPH 2: STREAM B (ANOMALY LOCALIZED WARPING)
# ==========================================
def save_stream_b():
    data_b = np.random.uniform(1.0, 1.8, size=(10, 10))
    # Inject localized severe strain concentration in the middle
    data_b[3:7, 3:7] = np.random.uniform(4.8, 5.8, size=(4, 4))

    plt.figure(figsize=(6, 5))
    im = plt.imshow(data_b, cmap='jet', extent=[0, 300, 0, 300], origin='lower')
    cbar = plt.colorbar(im)
    cbar.set_label('True Strain Amplitude ($\epsilon$)', fontsize=11)
    
    plt.title('Stream B: Anomaly Alert Layer Localized Warping Defect', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('X-Coordinate Spatial Grid (pixels)', fontsize=10)
    plt.ylabel('Y-Coordinate Spatial Grid (pixels)', fontsize=10)
    plt.tight_layout()
    plt.savefig('graph2_stream_b_anomaly.jpg', dpi=300, format='jpg')
    print(" Saved: graph2_stream_b_anomaly.jpg")
    plt.close()

# ==========================================
# GRAPH 3: RAW TELEMETRY SENSOR COMPARATIVES
# ==========================================
def save_telemetry_comparatives():
    labels = ['Nozzle Temp\n(°C)', 'Bed Temp\n(°C)', 'Print Speed\n(mm/s)', 'Infill Time\n(s)', 'Travel Time\n(s)', 'Dim. Dev\n(mm)']
    stream_a = [210.0, 60.0, 45.0, 220.0, 18.0, 0.02]
    stream_b = [195.0, 58.0, 65.0, 180.0, 12.0, 0.12]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width/2, stream_a, width, label='Stream A (Nominal)', color='#2E7D32')
    plt.bar(x + width/2, stream_b, width, label='Stream B (Anomaly)', color='#C62828')

    plt.yscale('log') # Log scale for magnitude differences
    plt.ylabel('Sensor Log values Amplitude Scale (Log Scale)', fontsize=10)
    plt.title('Raw Print Telemetry Sensor Input Streams Comparatives', fontsize=12, fontweight='bold', pad=10)
    plt.xticks(x, labels, fontsize=9)
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph3_telemetry_comparatives.jpg', dpi=300, format='jpg')
    print(" Saved: graph3_telemetry_comparatives.jpg")
    plt.close()

# ==========================================
# GRAPH 4: TRANSFORMER FUSED NETWORK OUTPUTS
# ==========================================
def save_transformer_confidence():
    categories = ['Nominal Layer Confidence', 'Anomaly Defect Confidence']
    stream_a_conf = [100, 0]
    stream_b_conf = [0, 100]

    plt.figure(figsize=(6, 5))
    plt.bar([0], [100], color='blue', width=0.4, label='Model Evaluation (Stream A)')
    plt.bar([1], [100], color='orange', width=0.4, label='Model Evaluation (Stream B)')

    plt.ylabel('Probability Distribution Percentage (%)', fontsize=10)
    plt.title('Cross-Modal Transformer Fused Network Outputs', fontsize=12, fontweight='bold', pad=10)
    plt.xticks([0, 1], categories, fontweight='bold', fontsize=10)
    plt.ylim(0, 110)
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.savefig('graph4_transformer_confidence.jpg', dpi=300, format='jpg')
    print(" Saved: graph4_transformer_confidence.jpg")
    plt.close()

if __name__ == "__main__":
    print("\nExtracting and saving individual dashboard graphs as JPGs...")
    save_stream_a()
    save_stream_b()
    save_telemetry_comparatives()
    save_transformer_confidence()
    print("All 4 graphs saved individually!\n")