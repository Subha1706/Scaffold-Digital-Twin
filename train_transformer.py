# train_transformer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# ==========================================
# PHASE 3: FEATURE EXTRACTION ARCHITECTURES
# ==========================================

class TabularCNNLSTMExtractor(nn.Module):
    """
    Processes sequential tabular print bed telemetry streams 
    to capture dynamic, high-frequency physical anomalies.
    """
    def __init__(self, input_dim=6, hidden_dim=32):
        super(TabularCNNLSTMExtractor, self).__init__()
        # 1D CNN layer to isolate local feature correlations across time steps
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=1)
        self.relu = nn.ReLU()
        # LSTM layer to track transient operational drift patterns
        self.lstm = nn.LSTM(input_size=16, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
    def forward(self, x):
        # Reshape for Conv1d: [Batch, Features, Sequence Length (1)]
        x = x.unsqueeze(-1)
        x = self.relu(self.conv1d(x))
        # Permute back for LSTM: [Batch, Sequence Length, Features]
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, _) = self.lstm(x)
        # Extract the final hidden state context vector
        return h_n[-1]

class StrainSpatialExtractor(nn.Module):
    """
    Processes 2D µDIC Hencky strain maps to extract 
    spatial topographical deformation gradients.
    """
    def __init__(self, hidden_dim=32):
        super(StrainSpatialExtractor, self).__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Process structural coordinates: [Batch, 1] -> [Batch, Hidden Dim]
        return self.spatial_mlp(x)

# ==========================================
# PHASE 4: MULTIMODAL TRANSFORMER FUSION
# ==========================================

class ScaffoldDigitalTwinTransformer(nn.Module):
    def __init__(self, tabular_dim=6, embed_dim=32, num_heads=4, num_classes=2):
        super(ScaffoldDigitalTwinTransformer, self).__init__()
        
        # Explicit Phase 3 Front-End Feature Extractors
        self.tabular_extractor = TabularCNNLSTMExtractor(input_dim=tabular_dim, hidden_dim=embed_dim)
        self.strain_extractor = StrainSpatialExtractor(hidden_dim=embed_dim)
        
        # Core Cross-Modal Attention Transformer block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=64, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x_tab, x_mech):
        # Extract features using our custom architectures
        tab_features = self.tabular_extractor(x_tab).unsqueeze(1)  # [B, 1, embed_dim]
        mech_features = self.strain_extractor(x_mech).unsqueeze(1) # [B, 1, embed_dim]
        
        # Merge tokens into a multi-modal sequence matrix
        multimodal_sequence = torch.cat((tab_features, mech_features), dim=1)
        
        # Compute cross-modal self-attention matrix paths
        fused_context = self.transformer_encoder(multimodal_sequence)
        fused_pooled = torch.mean(fused_context, dim=1)
        
        return self.classifier_head(fused_pooled)

if __name__ == "__main__":
    print("====================================================")
    print("    TRAINING COMPLIANT MULTIMODAL NETWORK (PHASE 3) ")
    print("====================================================")
    
    data_pack_path = "scaffold_twin_dataset.pt"
    if not os.path.exists(data_pack_path):
        raise FileNotFoundError("Missing tensor dataset registry file.")
        
    data_pack = torch.load(data_pack_path)
    dataset = TensorDataset(data_pack['tabular_features'], data_pack['mechanical_features'], data_pack['labels'])
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = ScaffoldDigitalTwinTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    print("[+] Feature extraction blocks initialized successfully. Running loops...")
    
    model.train()
    for epoch in range(1, 21):
        epoch_loss = 0.0
        for batch_tab, batch_mech, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_tab, batch_mech)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 5 == 0:
            print(f"    -> Epoch [{epoch:02d}/20] completed optimization profile.")
            
    torch.save(model.state_dict(), "scaffold_twin_transformer.pth")
    print("\n[✔] SUCCESS: Phase 3 Feature Extractors and Fusion Architecture Trained!")
    print("====================================================")