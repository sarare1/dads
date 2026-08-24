import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DBackbone(nn.Module):
    def __init__(self, input_dim: int = 6, embed_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * input_dim, embed_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))

class TransformerBackbone(nn.Module):
    def __init__(self, input_dim: int = 6, embed_dim: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(1, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(32 * input_dim, embed_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))

class HybridBackbone(nn.Module):
    def __init__(self, input_dim: int = 6, embed_dim: int = 128):
        super().__init__()
        self.cnn = CNN1DBackbone(input_dim, 64)
        self.transformer = TransformerBackbone(input_dim, 64)
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        out_cnn = self.cnn(x)
        out_tf = self.transformer(x)
        combined = torch.cat([out_cnn, out_tf], dim=-1)
        return F.relu(self.fc(combined))

class DualHeadEWModel(nn.Module):
    """
    SOSA-Aligned Dual-Head Neural Architecture:
    - Shared Backbone Extractor (1D-CNN / Transformer / Hybrid)
    - Branch A: Closed-World Classifier Head
    - Branch B: Open-World Metric Projection Head (L2 Normalized)
    """
    def __init__(self, backbone_type: str = "hybrid", input_dim: int = 6, 
                 embed_dim: int = 128, num_classes: int = 20):
        super().__init__()
        self.backbone_type = backbone_type
        if backbone_type == "cnn1d":
            self.backbone = CNN1DBackbone(input_dim, embed_dim)
        elif backbone_type == "transformer":
            self.backbone = TransformerBackbone(input_dim, embed_dim)
        else:
            self.backbone = HybridBackbone(input_dim, embed_dim)

        self.classifier_head = nn.Linear(embed_dim, num_classes)
        
        self.metric_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.classifier_head(features)
        probs = F.softmax(logits, dim=-1)
        
        embeddings = self.metric_head(features)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return probs, embeddings
