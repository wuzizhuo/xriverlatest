import torch
import torch.nn as nn


class SigREG(nn.Module):
    def __init__(self, *, input_dim: int, lam: float = 0.1, sketch_dim: int = 128, seed: int = 0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.lam = float(lam)
        self.sketch_dim = int(sketch_dim)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        mat = torch.randn(int(self.input_dim), int(self.sketch_dim), generator=gen, dtype=torch.float32) / float(max(1, int(self.input_dim)) ** 0.5)
        self.register_buffer("sketch_mat", mat, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if float(self.lam) <= 0.0:
            return torch.zeros((), device=z.device, dtype=torch.float32)
        if z.ndim != 2:
            z = z.view(int(z.size(0)), -1)
        b = int(z.size(0))
        if b < 2:
            return torch.zeros((), device=z.device, dtype=torch.float32)
        if int(z.size(1)) != int(self.input_dim):
            raise ValueError(f"SigREG input_dim mismatch: got {int(z.size(1))} expected {int(self.input_dim)}")
        m = self.sketch_mat.to(device=z.device, dtype=torch.float32)
        z_sketch = z.to(dtype=torch.float32) @ m
        mean = z_sketch.mean(dim=0)
        loss_mean = torch.sum(mean**2)
        z_c = z_sketch - mean
        cov = (z_c.T @ z_c) / float(max(1, b - 1))
        eye = torch.eye(int(self.sketch_dim), device=z.device, dtype=torch.float32)
        loss_cov = torch.sum((cov - eye) ** 2)
        return float(self.lam) * (loss_mean + loss_cov)
