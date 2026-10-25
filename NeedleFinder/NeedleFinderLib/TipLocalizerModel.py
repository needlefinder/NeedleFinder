"""Training architecture shared by paper experiments and model export."""
import torch
from torch import nn
from torch.nn import functional as F

class Dedicated1DTipLocalizer(nn.Module):
    """
    Dedicated 1D Multi-Channel Axial Tip Localizer operating on 91-step (0.5 mm)
    22-channel profiles [B, 22, 91] from z_top - 25.0 mm to z_top + 20.0 mm.
    Combines multi-scale dilated 1D ResNet blocks with 1D Axial Self-Attention and
    predicts both a 91-step tip probability distribution (soft-argmax over [-25, +20] mm)
    and a direct residual refinement.
    """
    def __init__(self, in_channels=22, hidden_dim=48, seq_len=91):
        super().__init__()
        self.register_buffer("dz_grid", torch.linspace(-25.0, 20.0, seq_len))
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, dilation=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, dilation=1),
            nn.GroupNorm(8, hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(8, hidden_dim),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=8, dilation=4),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=8, dilation=4),
            nn.GroupNorm(8, hidden_dim),
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.05, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.step_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: [B, 22, 91]
        h = self.stem(x)
        h = F.silu(h + self.block1(h))
        h = F.silu(h + self.block2(h))
        h = F.silu(h + self.block3(h))

        # 1D self-attention across the 91 axial steps
        h_seq = h.transpose(1, 2)  # [B, 91, C]
        attn_out, _ = self.attn(h_seq, h_seq, h_seq, need_weights=False)
        h_seq = self.attn_norm(h_seq + attn_out)
        h = h_seq.transpose(1, 2)  # [B, C, 91]

        step_logits = self.step_head(h).squeeze(1)  # [B, 91]
        step_probs = F.softmax(step_logits, dim=-1)
        dz_soft = torch.sum(step_probs * self.dz_grid.unsqueeze(0), dim=-1)  # [B]

        # Global + center context for residual regression
        h_center = h[:, :, 50]  # index 50 is dz = 0.0 mm
        h_mean = h.mean(dim=-1)
        dz_res = self.reg_head(torch.cat([h_center, h_mean], dim=-1)).squeeze(-1) * 10.0
        dz_pred_mm = 0.75 * dz_soft + 0.25 * dz_res
        return dz_pred_mm, step_logits
