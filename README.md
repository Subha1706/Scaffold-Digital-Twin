# A Multimodal Cross-Attention Digital Twin for Adaptive Defect Detection in 3D-Printed Scaffolds Using DIC Feedback

## 📌 Overview

This project presents a **multimodal Digital Twin framework for real-time defect detection and adaptive control in 3D-printed tissue-engineering scaffolds**.

The system combines:

* **6-DoF machine process telemetry**
* **Microscopic optical images**
* **Digital Image Correlation (μDIC) strain fields**
* **Deep learning-based multimodal fusion**
* **Unsupervised anomaly detection**
* **Closed-loop G-Code control**

Unlike conventional post-process inspection systems, the proposed Digital Twin continuously monitors the printing process and can **detect defects while printing and automatically adjust printer parameters before defects propagate**.

The core architecture uses a **4-head Cross-Modal Transformer** to fuse temporal machine telemetry with spatial μDIC strain information. An **Unsupervised Latent Autoencoder** acts as a safety mechanism for detecting previously unseen anomalies.

---

## 🎯 Problem Statement

3D-printed tissue-engineering scaffolds require precise microstructural properties to maintain mechanical strength, pore connectivity, and biological functionality.

During extrusion-based printing, defects such as:

* Interlayer delamination
* Warping
* Under-extrusion
* Thermal instability
* Strand bridging failure
* Localized deformation

can occur because of changes in nozzle temperature, printing speed, cooling, and material behavior.

Traditional quality inspection approaches have several limitations:

1. **Post-process inspection**
   Defects may only be detected after the printing process is completed, resulting in material and time loss.

2. **Single-sensor limitations**
   Conventional cameras may detect visible geometric defects but cannot directly capture localized strain or thermal/process abnormalities.

3. **Passive Digital Twins**
   Many Digital Twin systems primarily monitor and visualize process information without actively modifying printer parameters.

This project addresses these limitations using a **multimodal, closed-loop Digital Twin architecture**.

---

# 🚀 Proposed Solution

The proposed system continuously combines information from multiple sources:

```text
                ┌─────────────────────────┐
                │   3D Printing Process   │
                └────────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       Optical Images   μDIC Strain    Machine Telemetry
         300 × 300       10 × 10          6-DoF
              │              │              │
              │              │              │
              ▼              ▼              ▼
        Image/Spatial     Spatial MLP    CNN-LSTM
         Processing       Embedding      Embedding
              │              │              │
              └───────┬──────┘
                      │
                      ▼
          ┌────────────────────────┐
          │ Cross-Modal Transformer│
          │       4 Attention Heads│
          └────────────┬───────────┘
                       │
                       ▼
              Defect Probability
                       │
                       ├───────────────┐
                       │               │
                       ▼               ▼
              Classification      Autoencoder
                                    Safety Net
                       │               │
                       └───────┬───────┘
                               ▼
                       Risk Severity γ
                               │
                               ▼
                  Closed-Loop Controller
                               │
                               ▼
                         G-Code Update
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              M104 Temperature      M220 Feed Rate
                    │                     │
                    └──────────┬──────────┘
                               ▼
                         3D Printer
```

The system therefore forms a **closed cyber-physical feedback loop**:

**Sense → Analyze → Detect → Decide → Act → Monitor**

---

# 🧠 System Architecture

## 1. Multimodal Input

The system uses three synchronized data streams.

### A. Microscopic Optical Images

Layer-wise images of the printed scaffold are captured at:

```text
300 × 300 pixels
```

These images provide information about strand morphology and visible printing defects.

### B. μDIC Strain Fields

Digital Image Correlation is used to estimate deformation across the printed surface.

The strain field is represented using a:

```text
10 × 10 spatial mesh
```

The system calculates deformation information including **True Hencky strain**.

### C. 6-DoF Process Telemetry

The machine telemetry contains six process variables:

| Feature | Description          |
| ------- | -------------------- |
| `T_n`   | Nozzle Temperature   |
| `T_b`   | Bed Temperature      |
| `v_p`   | Print Speed          |
| `v_t`   | Travel Speed         |
| `t_i`   | Infill Time          |
| `δ_d`   | Dimensional Variance |

These streams are synchronized at the layer level before being provided to the neural network.

---

# 📐 μDIC Strain Processing

The Digital Image Correlation pipeline tracks displacement fields:

```text
u(x,y)
v(x,y)
```

from sequential images.

The displacement fields are transformed into strain tensors using Green-Lagrange strain formulations.

The True Hencky logarithmic strain is then calculated as:

```text
ε_H = 1/2 ln(1 + 2E_ij)
```

This allows the system to identify localized deformation and strain concentrations that may indicate the early stages of structural defects.

---

# 🔄 Data Preprocessing

Because the different sensor channels have very different numerical scales, the system performs **Z-score normalization**.

The normalization is:

```text
z_i = (x_i - μ_i) / σ_i
```

where:

* `μ_i` = mean of the feature
* `σ_i` = standard deviation of the feature

The synchronized multimodal data is stored in:

```text
scaffold_twin_dataset.pt
```

The preprocessing produces aligned layer-wise representations suitable for multimodal deep learning.

---

# 🧩 Feature Extraction

The architecture converts the different modalities into a common:

```text
32-dimensional embedding space
```

## Temporal Feature Extractor — CNN + LSTM

The six telemetry features are processed using a hybrid:

```text
1D CNN → LSTM
```

The CNN captures local patterns in the telemetry sequence, while the LSTM captures temporal dependencies between consecutive printing states.

The resulting temporal embedding is:

```text
e_temp ∈ R^32
```

### CNN Configuration

```text
Filters       : 16
Kernel size   : 3
Stride        : 1
Activation    : ReLU
```

### LSTM

```text
Hidden dimension : 32
```

---

# 🗺️ Spatial Feature Extractor — Spatial MLP

The:

```text
10 × 10
```

strain mesh is flattened into a:

```text
100-dimensional vector
```

and passed through a dense bottleneck:

```text
100 → 16 → 32
```

The final spatial representation is:

```text
e_spat ∈ R^32
```

This converts the spatial strain distribution into the same embedding dimension as the telemetry representation.

---

# 🤖 Cross-Modal Transformer

The core of the proposed architecture is a **4-head Cross-Modal Transformer**.

Instead of simply concatenating the telemetry and strain features, the model learns relationships between the two modalities using cross-attention.

The representations are:

```text
Temporal Token → Query (Q)

Spatial Token → Key (K)
Spatial Token → Value (V)
```

The attention mechanism is:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √d_k)V
```

where:

```text
d_k = 8
```

for each attention head.

The resulting representation is combined with the temporal embedding using residual connection and Layer Normalization.

Finally, the fused representation is passed to a classification head that produces:

```text
P_defect ∈ [0,1]
```

representing the probability of a defect.

---

# 🛡️ Unsupervised Anomaly Detection

A major challenge in manufacturing is that it is impossible to collect labeled examples for every possible defect.

To address this problem, the project introduces an **Unsupervised Latent Autoencoder**.

The Autoencoder is trained only on:

```text
Healthy / Nominal Printing Layers
```

Architecture:

```text
32 → 8 → 32
```

The model learns to reconstruct normal printing behavior.

The reconstruction error is calculated using:

```text
MSE = (1/N) Σ(X_j - X̂_j)²
```

A calibrated anomaly threshold is:

```text
MSE = 0.170
```

If:

```text
MSE > 0.170
```

the system considers the observation anomalous.

This provides a safety mechanism for detecting defects that were not present in the supervised training data.

---

# ⚙️ Closed-Loop Adaptive Control

The system does not stop after detecting a defect.

When an anomaly is detected, the controller calculates a composite risk score:

```text
γ ∈ [0,1]
```

using:

```text
γ = max(
        P_defect,
        min(MSE_rec / (2 × 0.170), 1.0)
    )
```

The controller then dynamically adjusts the printing parameters.

## Temperature Adjustment

Baseline:

```text
T_nom = 210°C
```

Target temperature:

```text
T_target = T_nom + (γ × 5°C)
```

Therefore, the maximum thermal adjustment is:

```text
+5°C
```

---

## Feed Rate Adjustment

Baseline:

```text
F_nom = 1.0
```

Target feed rate:

```text
F_target = max(0.70, F_nom - γ × 0.25)
```

This prevents the feed rate from falling below:

```text
70%
```

---

# 🖨️ Automatic G-Code Generation

The controller converts the calculated parameters into printer commands.

### Temperature Control

```text
M104 S[temp]
```

### Feed Rate Control

```text
M220 S[feed]
```

For example:

```text
M104 S214.4
M220 S78
```

This enables the Digital Twin to respond to defects during active printing rather than waiting for post-process inspection.

The complete control logic is implemented through:

```text
closed_loop_twin.py
```

The overall control pipeline is:

```text
Sensor Data
     ↓
Normalization
     ↓
CNN-LSTM + Spatial MLP
     ↓
Cross-Modal Transformer
     ↓
Defect Probability
     ↓
Autoencoder Reconstruction Error
     ↓
Risk Severity γ
     ↓
Parameter Adjustment
     ↓
G-Code Generation
     ↓
Printer Firmware
```

---

# 📊 Experimental Results

## Classification Performance

The proposed Cross-Modal Transformer achieved:

```text
86.0% classification accuracy
```

on the 100-layer evaluation set.

The confusion matrix contained:

```text
44 True Nominal classifications
42 True Defect classifications
```

The system therefore demonstrated the ability to distinguish normal and defective printing conditions using multimodal information.

---

# 🔬 Ablation Study

The contribution of multimodal fusion and cross-attention was evaluated using different architectures.

| Architecture                               |  Accuracy | Tripwire Recall |
| ------------------------------------------ | --------: | --------------: |
| 1D Telemetry Only — CNN-LSTM               |     68.4% |             N/A |
| Spatial Strain Mesh Only — Spatial MLP     |     74.2% |             N/A |
| Multimodal Concatenation without Attention |     79.1% |           81.0% |
| **Cross-Attention + Autoencoder**          | **86.0%** |       **94.2%** |

The results show that combining temporal telemetry and spatial strain information through cross-attention provides a stronger representation than using either modality independently or simply concatenating their features.

---

# 🔎 Explainability

The project also analyzes the internal attention weights of the Cross-Modal Transformer.

During normal printing, the attention distribution remains relatively uniform across the telemetry features.

During defect formation, the attention becomes concentrated on important process variables.

For the analyzed defective stream:

```text
Nozzle Temperature      → 38.2%
Dimensional Deviation   → 34.6%
```

This provides an interpretable indication of which process variables contributed strongly to the detected anomaly.

---

# 🔁 Closed-Loop Validation

The controller was tested under three representative scenarios.

## Scenario 1 — Nominal Layer

```text
P_defect = 0.12
MSE      = 0.0004
```

No corrective action is required.

Generated command:

```text
M104 S210.0
M220 S100
```

---

## Scenario 2 — Known Defect

```text
P_defect = 0.88
MSE      = 0.0004
```

Since:

```text
P_defect > 0.75
```

the supervised defect detector triggers the controller.

Generated command:

```text
M104 S214.4
M220 S78
```

This corresponds to:

```text
+4.4°C temperature adjustment
22% feed-rate reduction
```

---

## Scenario 3 — Unseen Anomaly

The supervised classifier does not identify the anomaly:

```text
P_defect = 0.12
```

However, the Autoencoder detects abnormal reconstruction:

```text
MSE = 0.2500
```

Since:

```text
0.2500 > 0.170
```

the unsupervised safety mechanism triggers.

Generated command:

```text
M104 S213.7
M220 S82
```

This demonstrates how the Autoencoder can provide an additional safety layer for previously unseen defect patterns.

---

# ⚡ Real-Time Performance

The system was benchmarked over:

```text
5,000 continuous inference cycles
```

Performance:

| Metric                         |           Result |
| ------------------------------ | ---------------: |
| Inference Steps                |            5,000 |
| Mean Processing Latency        | 1.014 ± 0.082 ms |
| Peak Throughput                |  986.3 ± 12.4 Hz |
| Industrial Real-Time Threshold |          < 50 ms |
| Real-Time Compliance           |           PASSED |

The measured processing latency of approximately:

```text
1.014 ms
```

is substantially below the specified 50 ms real-time threshold.

---

# 🧪 Experimental Setup

The experimental system consisted of an extrusion-based additive manufacturing setup with:

```text
Nozzle Diameter       : 0.4 mm
Material              : Medical-grade PCL
Baseline Print Speed  : 45 mm/s
Ambient Temperature   : 24.0 ± 0.5°C
```

The optical system used a high-magnification telecentric camera with cross-polarized ring lighting.

The Digital Twin was executed on:

```text
CPU : Intel Core i7, 2.8 GHz
RAM : 16 GB
GPU : NVIDIA RTX GPU
```

Machine telemetry was transmitted using asynchronous serial communication at:

```text
115200 bps
```

The experimental configuration is described in the research paper.

---

# 🛠️ Technologies Used

## Machine Learning / Deep Learning

* Python
* PyTorch
* 1D CNN
* LSTM
* Multi-Head Cross-Attention
* Transformer Architecture
* Autoencoder
* Z-score normalization

## Computer Vision / Image Processing

* Digital Image Correlation (DIC)
* μDIC
* Optical deformation analysis
* Strain field processing

## Additive Manufacturing

* FDM / Extrusion-based 3D Printing
* PCL filament
* G-Code
* Marlin/Duet-style printer commands

## Cyber-Physical System

* Digital Twin
* Real-time telemetry
* Serial communication
* Closed-loop control

---

# 📁 Project Structure

A suggested repository structure is:

```text
.
├── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── scaffold_twin_dataset.pt
│
├── models/
│   ├── cnn_lstm.py
│   ├── spatial_mlp.py
│   ├── cross_modal_transformer.py
│   └── autoencoder.py
│
├── training/
│   ├── train_classifier.py
│   └── train_autoencoder.py
│
├── inference/
│   └── inference.py
│
├── control/
│   └── closed_loop_twin.py
│
├── dic/
│   ├── strain_processing.py
│   └── dic_utils.py
│
├── visualization/
│   ├── plot_strain.py
│   ├── plot_telemetry.py
│   └── plot_confusion_matrix.py
│
├── results/
│   ├── confusion_matrix/
│   ├── strain_maps/
│   └── latency/
│
└── requirements.txt
```

---

# 🔄 Complete Workflow

```text
1. Acquire Printing Data
          ↓
2. Capture Optical Images
          ↓
3. Calculate μDIC Displacement
          ↓
4. Calculate Strain Fields
          ↓
5. Collect 6-DoF Machine Telemetry
          ↓
6. Synchronize Multimodal Data
          ↓
7. Apply Z-Score Normalization
          ↓
8. Extract Temporal Features
   CNN → LSTM
          ↓
9. Extract Spatial Features
   Spatial MLP
          ↓
10. Generate 32-Dimensional Tokens
          ↓
11. Cross-Modal Transformer
          ↓
12. Predict Defect Probability
          ↓
13. Autoencoder Anomaly Detection
          ↓
14. Calculate Risk Severity
          ↓
15. Generate G-Code Override
          ↓
16. Send Command to Printer
          ↓
17. Continue Monitoring
```

---

# 💡 Key Contributions

The project provides the following main contributions:

### 1. Multimodal Digital Twin

Combines:

```text
Machine Telemetry + μDIC Strain
```

within a unified Digital Twin architecture.

### 2. Cross-Modal Attention

Uses a:

```text
4-head Cross-Modal Transformer
```

to learn relationships between process telemetry and spatial deformation.

### 3. Unsupervised Safety Mechanism

Uses a latent Autoencoder trained on nominal printing states to detect unseen anomalies.

### 4. Closed-Loop Control

Converts anomaly detection results directly into printer parameter adjustments using:

```text
M104
M220
```

G-Code commands.

### 5. Low-Latency Inference

The complete inference and control pipeline achieved:

```text
1.014 ± 0.082 ms
```

mean processing latency.

---

# 📈 Key Results

| Metric                           |               Result |
| -------------------------------- | -------------------: |
| Cross-Modal Transformer Accuracy |            **86.0%** |
| Tripwire Recall                  |            **94.2%** |
| Autoencoder Threshold            |        **0.170 MSE** |
| Mean Inference Latency           | **1.014 ± 0.082 ms** |
| Peak Throughput                  |  **986.3 ± 12.4 Hz** |
| Evaluation Samples               |       **100 layers** |
| Continuous Benchmark Cycles      |            **5,000** |

---

# 🔮 Future Work

The research identifies several directions for extending the system:

* Hardware-in-the-loop (HIL) deployment
* ROS2 integration
* MQTT-based communication
* Grad-CAM-based explainability
* Extension from 2D strain fields to 3D volumetric strain monitoring
* Graph Neural Networks (GNNs) for spatial modeling

These extensions would allow the Digital Twin to move toward more complete physical integration and three-dimensional defect monitoring.

---

# 📚 References

1. F. Tao, H. Zhang, A. Liu, and A. Y. Nee, "Digital twin in industry: State-of-the-art," IEEE Transactions on Industrial Informatics, 2019.

2. C. Liu, Z. Le, and D. Adams, "Machine learning for in-situ quality inspection in additive manufacturing," Journal of Manufacturing Systems, 2021.

3. M. A. Sutton, J. J. Orteu, and H. Schreier, *Image Correlation for Shape, Motion and Deformation Measurements*, Springer, 2009.

4. P. L. Reu et al., "DIC Challenge 2.0: Developing Images and Guidelines for Evaluating Accuracy and Resolution of 2D Analyses," Experimental Mechanics, 2022.

5. Y. Wang, Z. Zhang, and X. Lin, "Multimodal sensor fusion for real-time location-dependent defect detection in additive manufacturing," IEEE Transactions on Industrial Informatics, 2023.

6. Z. Jin, Z. Zhang, and G. X. Gu, "Autonomous in-situ defect detection in additive manufacturing via deep learning," Materials & Design, 2020.

7. A. Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.

8. I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*, MIT Press, 2016.

---

# 👩‍💻 Authors

**Y. Subhasree**
Department of Computer Science and Engineering
IIITDM Kancheepuram, Chennai, India

**Rajesh Kolluri**
Department of Computer Science and Engineering
IIITDM Kancheepuram, Chennai, India

---

# ⭐ Project Summary

> **A multimodal Digital Twin that combines machine telemetry and μDIC strain fields using Cross-Modal Attention and an unsupervised anomaly detection safety net to detect defects and adapt 3D-printing parameters in real time.**

The project demonstrates a complete **Sense → Detect → Decide → Act** pipeline for intelligent additive manufacturing and provides a foundation for future hardware-in-the-loop Digital Twin systems.
