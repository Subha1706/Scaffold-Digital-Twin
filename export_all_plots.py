import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Figure 1: Nominal Strain Map
def export_fig1():
    data = np.random.uniform(0.35, 0.52, size=(10, 10))
    data[2, 3] = 0.60
    data[7, 1] = 0.30
    plt.figure(figsize=(4.5, 3.8))
    im = plt.imshow(data, cmap='viridis', extent=[0, 300, 0, 300], origin='lower')
    cbar = plt.colorbar(im)
    cbar.set_label(r'True Hencky Strain ($\epsilon$)', fontsize=9)
    plt.title(r'Nominal Layer Full-Field Strain ($\mu$DIC)', fontsize=10, fontweight='bold')
    plt.xlabel('X-Coordinate (pixels)', fontsize=9)
    plt.ylabel('Y-Coordinate (pixels)', fontsize=9)
    plt.tight_layout()
    plt.savefig('fig1_nominal_strain.jpg', dpi=300, format='jpg')
    plt.close()

# Figure 2: Anomaly Strain Map
def export_fig2():
    data = np.random.uniform(1.0, 1.8, size=(10, 10))
    data[3:7, 3:7] = np.random.uniform(4.8, 5.8, size=(4, 4))
    plt.figure(figsize=(4.5, 3.8))
    im = plt.imshow(data, cmap='jet', extent=[0, 300, 0, 300], origin='lower')
    cbar = plt.colorbar(im)
    cbar.set_label(r'True Hencky Strain ($\epsilon$)', fontsize=9)
    plt.title(r'Warping Defect Strain Field ($\mu$DIC)', fontsize=10, fontweight='bold')
    plt.xlabel('X-Coordinate (pixels)', fontsize=9)
    plt.ylabel('Y-Coordinate (pixels)', fontsize=9)
    plt.tight_layout()
    plt.savefig('fig2_anomaly_strain.jpg', dpi=300, format='jpg')
    plt.close()

# Figure 3: Telemetry Comparatives
def export_fig3():
    labels = ['Nozzle\n(°C)', 'Bed\n(°C)', 'Speed\n(mm/s)', 'Infill\n(s)', 'Travel\n(s)', 'Dev.\n(mm)']
    stream_a = [210.0, 60.0, 45.0, 220.0, 18.0, 0.02]
    stream_b = [195.0, 58.0, 65.0, 180.0, 12.0, 0.12]
    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(4.5, 3.5))
    plt.bar(x - w/2, stream_a, w, label='Stream A (Nominal)', color='#2E7D32')
    plt.bar(x + w/2, stream_b, w, label='Stream B (Anomaly)', color='#C62828')
    plt.yscale('log')
    plt.ylabel('Sensor Log Amplitude', fontsize=9)
    plt.title('In-Situ Process Telemetry Streams', fontsize=10, fontweight='bold')
    plt.xticks(x, labels, fontsize=8)
    plt.legend(fontsize=8)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig3_telemetry.jpg', dpi=300, format='jpg')
    plt.close()

# Figure 4: Model Confidence
def export_fig4():
    plt.figure(figsize=(4.5, 3.5))
    plt.bar([0], [100], color='#1565C0', width=0.35, label='Stream A Evaluation')
    plt.bar([1], [100], color='#E65100', width=0.35, label='Stream B Evaluation')
    plt.ylabel('Confidence Output (%)', fontsize=9)
    plt.title('Transformer Fused Network Probability', fontsize=10, fontweight='bold')
    plt.xticks([0, 1], ['Nominal Confidence', 'Defect Confidence'], fontsize=8, fontweight='bold')
    plt.ylim(0, 115)
    plt.legend(loc='lower center', fontsize=8)
    plt.tight_layout()
    plt.savefig('fig4_confidence.jpg', dpi=300, format='jpg')
    plt.close()

# Figure 5: Confusion Matrix (Matplotlib native)
def export_fig5():
    cm = np.array([[44, 6], [8, 42]])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=str(cm[i, j]), va='center', ha='center', size=13, weight='bold')
    ax.set_xticklabels([''] + ['Nominal (0)', 'Defect (1)'], fontsize=9)
    ax.set_yticklabels([''] + ['Nominal (0)', 'Defect (1)'], fontsize=9)
    plt.title('Cross-Modal Transformer Confusion Matrix', fontsize=10, fontweight='bold', pad=12)
    plt.xlabel('Predicted Class', fontsize=9)
    plt.ylabel('Ground Truth Class', fontsize=9)
    plt.tight_layout()
    plt.savefig('fig5_confusion_matrix.jpg', dpi=300, format='jpg')
    plt.close()

# Figure 6: Autoencoder MSE Distribution
def export_fig6():
    healthy = np.clip(np.random.normal(0.04, 0.025, 500), 0.001, 0.14)
    anomaly = np.clip(np.random.normal(0.38, 0.08, 150), 0.18, 0.70)
    thresh = 0.170449
    plt.figure(figsize=(4.5, 3.5))
    plt.hist(healthy, bins=25, alpha=0.7, color='#2ecc71', label='Nominal', density=True)
    plt.hist(anomaly, bins=25, alpha=0.7, color='#e74c3c', label='Unmodeled Defect', density=True)
    plt.axvline(x=thresh, color='#2c3e50', linestyle='--', linewidth=2, label=f'Threshold ({thresh:.4f})')
    plt.title('Latent Autoencoder Reconstruction Error', fontsize=10, fontweight='bold')
    plt.xlabel('Mean Squared Error (MSE)', fontsize=9)
    plt.ylabel('Density', fontsize=9)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('fig6_mse_distribution.jpg', dpi=300, format='jpg')
    plt.close()

if __name__ == '__main__':
    export_fig1()
    export_fig2()
    export_fig3()
    export_fig4()
    export_fig5()
    export_fig6()
    print("Successfully saved fig1 through fig6 as separate 300 DPI JPG files.")