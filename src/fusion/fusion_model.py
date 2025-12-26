# src/fusion/fusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityGate(nn.Module):
    def __init__(self, dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        gate = self.net(x)
        return x * gate

class FusionEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.net(x)

class MultimodalFusionNet(nn.Module):
    def __init__(self, modality_dims: dict, encoder_dim=128, trunk_dim=256):
        super().__init__()
        self.modalities = list(modality_dims.keys())
        self.encoders = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        self.encoder_dim = encoder_dim

        for k, d in modality_dims.items():
            self.encoders[k] = FusionEncoder(d, encoder_dim)
            self.gates[k] = ModalityGate(encoder_dim, hidden=max(16, encoder_dim // 4))

        total_dim = encoder_dim * len(self.modalities)

        self.cross = nn.Sequential(
            nn.Linear(total_dim, trunk_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU()
        )

        self.head_cognitive = nn.Sequential(nn.Linear(trunk_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_attention = nn.Sequential(nn.Linear(trunk_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_fatigue = nn.Sequential(nn.Linear(trunk_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_stress = nn.Sequential(nn.Linear(trunk_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_dict):
        """
        x_dict: {mod_name: tensor(B, dim)}
        All tensors are cast to same device & dtype as model weights.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        encoded = []
        batch_size = None

        for k in self.modalities:
            if k not in x_dict:
                raise KeyError(f"Missing modality '{k}' in x_dict")

            x = x_dict[k]
            if batch_size is None:
                batch_size = x.shape[0]

            # ensure correct device + dtype
            x = x.to(device=device, dtype=dtype)

            z = self.encoders[k](x)
            z = self.gates[k](z)
            encoded.append(z)

        cat = torch.cat(encoded, dim=1)
        t = self.cross(cat)

        c = torch.sigmoid(self.head_cognitive(t)).squeeze(-1)
        a = torch.sigmoid(self.head_attention(t)).squeeze(-1)
        f = torch.sigmoid(self.head_fatigue(t)).squeeze(-1)
        s = torch.sigmoid(self.head_stress(t)).squeeze(-1)

        return {
            "cognitive": c,
            "attention": a,
            "fatigue": f,
            "stress": s
        }
