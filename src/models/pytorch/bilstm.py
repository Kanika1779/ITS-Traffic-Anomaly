import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Focal Loss ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


# ── 2. Single BiLSTM Stream ───────────────────────────────────────────────────
class BiLSTMStream(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(BiLSTMStream, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.dropout(out)
        return out  # shape: (batch, seq_len, hidden_size*2)


# ── 3. Multi-Head Cross-Stream Attention ──────────────────────────────────────
class CrossStreamAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super(CrossStreamAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, temporal, spatial):
        # Cross attention: temporal queries spatial
        attn_out, attn_weights = self.attention(
            query=temporal,
            key=spatial,
            value=spatial
        )
        # Residual connection
        out = self.norm(temporal + self.dropout(attn_out))
        return out, attn_weights


# ── 4. AGTSF-Net (Full Model) ─────────────────────────────────────────────────
class AGTSFNet(nn.Module):
    def __init__(
        self,
        temporal_input_size=8,
        spatial_input_size=8,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        dropout=0.3
    ):
        super(AGTSFNet, self).__init__()

        # Dual Stream BiLSTM
        self.temporal_stream = BiLSTMStream(
            temporal_input_size, hidden_size, num_layers, dropout
        )
        self.spatial_stream = BiLSTMStream(
            spatial_input_size, hidden_size, num_layers, dropout
        )

        # Cross-Stream Attention (hidden_size*2 because bidirectional)
        embed_dim = hidden_size * 2
        self.cross_attention = CrossStreamAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Binary: normal or anomaly
        )

    def forward(self, x_temporal, x_spatial):
        # Pass through both BiLSTM streams
        temporal_out = self.temporal_stream(x_temporal)
        spatial_out  = self.spatial_stream(x_spatial)

        # Cross-stream attention fusion
        fused, attn_weights = self.cross_attention(temporal_out, spatial_out)

        # Take last timestep for classification
        last = fused[:, -1, :]

        # Classify
        logits = self.classifier(last)
        return logits.squeeze(-1), attn_weights


# ── 5. Quick Test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dummy input (batch=32, seq_len=60, features=8)
    x_temporal = torch.randn(32, 60, 8).to(device)
    x_spatial  = torch.randn(32, 60, 8).to(device)

    model = AGTSFNet().to(device)
    logits, attn = model(x_temporal, x_spatial)

    print(f"Output shape: {logits.shape}")
    print(f"Attention shape: {attn.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ AGTSF-Net built successfully!")