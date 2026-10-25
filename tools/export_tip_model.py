#!/usr/bin/env python3
"""Export the actual trained five-tap architecture; never retrain during export."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / 'NeedleFinder/NeedleFinderLib'
spec = importlib.util.spec_from_file_location('tip_model', LIB / 'TipLocalizerModel.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ExportModel(module.Dedicated1DTipLocalizer):
    def forward(self, x):
        dz, logits = super().forward(x)
        probs = torch.softmax(logits, dim=-1)
        mean = (probs * self.dz_grid).sum(dim=-1)
        std = ((probs * (self.dz_grid - mean[:, None])**2).sum(dim=-1).clamp(min=0.25)).sqrt()
        return dz.clamp(-25, 20), probs, std


def load_model():
    with np.load(LIB / 'needle_graphnet.npz', allow_pickle=False) as archive:
        weights = {k.removeprefix('tip1d.'): torch.from_numpy(archive[k].copy())
                   for k in archive.files if k.startswith('tip1d.')}
    model = ExportModel(in_channels=weights['stem.0.weight'].shape[1]).eval()
    model.load_state_dict(weights, strict=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    model = load_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.randn(2,24,91), str(args.output), opset_version=17,
                      input_names=['profile_24x91'], output_names=['dz_pred_mm','step_probs','step_std_mm'],
                      dynamic_axes={k:{0:'num_needles'} for k in ['profile_24x91','dz_pred_mm','step_probs','step_std_mm']})

if __name__ == '__main__': main()
