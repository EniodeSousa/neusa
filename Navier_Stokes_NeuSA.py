%%writefile Navier_Stokes_LBFGS.py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, hidden_dim=64, num_layers=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def tg_vortex_analytic(x, y, t, nu=0.01):
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=x.device, dtype=x.dtype)
    decay = torch.exp(-2.0 * nu * (torch.pi**2) * t)
    u_tgt = -torch.cos(torch.pi * x) * torch.sin(torch.pi * y) * decay
    v_tgt = torch.sin(torch.pi * x) * torch.cos(torch.pi * y) * decay
    return u_tgt, v_tgt

def compute_navier_stokes_loss(model, x, y, t, nu=0.01):
    inputs = torch.cat([x, y, t], dim=1)
    outputs = model(inputs)
    u, v = outputs[:, 0:1], outputs[:, 1:2]

    u_g = torch.autograd.grad(u, [x, y, t], torch.ones_like(u), create_graph=True)
    u_x, u_y, u_t = u_g[0], u_g[1], u_g[2]
    v_g = torch.autograd.grad(v, [x, y, t], torch.ones_like(v), create_graph=True)
    v_x, v_y, v_t = v_g[0], v_g[1], v_g[2]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

    res_continuity = u_x + v_y
    res_mom_u = u_t + (u * u_x) + (v * u_y) - nu * (u_xx + u_yy)
    res_mom_v = v_t + (u * v_x) + (v * v_y) - nu * (v_xx + v_yy)

    return torch.mean(res_continuity**2) + torch.mean(res_mom_u**2) + torch.mean(res_mom_v**2)

def plot_verification_snapshots(model, device, nu, save_path="results/high_precision_tgv.png"):
    model.eval()
    nx, ny = 100, 100
    L = 1.0
    x_lin = torch.linspace(-L, L, nx)
    y_lin = torch.linspace(-L, L, ny)
    grid_x, grid_y = torch.meshgrid(x_lin, y_lin, indexing="ij")
    
    time_snaps = [0.25, 0.50, 0.75, 1.00]
    fig, axes = plt.subplots(6, 4, figsize=(16, 18))

    with torch.no_grad():
        for i, t_val in enumerate(time_snaps):
            x_flat = grid_x.reshape(-1, 1).to(device)
            y_flat = grid_y.reshape(-1, 1).to(device)
            t_flat = torch.full_like(x_flat, t_val).to(device)
            inputs = torch.cat([x_flat, y_flat, t_flat], dim=1)

            u_tgt, v_tgt = tg_vortex_analytic(x_flat, y_flat, t_val, nu)
            U_tgt, V_tgt = u_tgt.reshape(nx, ny).cpu().numpy(), v_tgt.reshape(nx, ny).cpu().numpy()

            outputs = model(inputs)
            U_pred, V_pred = outputs[:, 0:1].reshape(nx, ny).cpu().numpy(), outputs[:, 1:2].reshape(nx, ny).cpu().numpy()

            U_err, V_err = np.abs(U_tgt - U_pred), np.abs(V_tgt - V_pred)

            im = axes[0, i].imshow(U_tgt, extent=[-L, L, -L, L], origin="lower", cmap="bwr", vmin=-1.0, vmax=1.0)
            axes[0, i].set_title(f"Target u | t={t_val}s"); fig.colorbar(im, ax=axes[0, i])
            
            im = axes[1, i].imshow(U_pred, extent=[-L, L, -L, L], origin="lower", cmap="bwr", vmin=-1.0, vmax=1.0)
            axes[1, i].set_title(f"Predicted u | t={t_val}s"); fig.colorbar(im, ax=axes[1, i])

            # Colorbar tightened back to 1e-4 to 1e-1 to show precision improvements
            im = axes[2, i].imshow(U_err, extent=[-L, L, -L, L], origin="lower", cmap="plasma", norm=LogNorm(vmin=1e-4, vmax=1e-1))
            axes[2, i].set_title(f"u Error"); fig.colorbar(im, ax=axes[2, i])

            im = axes[3, i].imshow(V_tgt, extent=[-L, L, -L, L], origin="lower", cmap="bwr", vmin=-1.0, vmax=1.0)
            axes[3, i].set_title(f"Target v | t={t_val}s"); fig.colorbar(im, ax=axes[3, i])

            im = axes[4, i].imshow(V_pred, extent=[-L, L, -L, L], origin="lower", cmap="bwr", vmin=-1.0, vmax=1.0)
            axes[4, i].set_title(f"Predicted v | t={t_val}s"); fig.colorbar(im, ax=axes[4, i])

            im = axes[5, i].imshow(V_err, extent=[-L, L, -L, L], origin="lower", cmap="plasma", norm=LogNorm(vmin=1e-4, vmax=1e-1))
            axes[5, i].set_title(f"v Error"); fig.colorbar(im, ax=axes[5, i])

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"High-precision plot saved to {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nu = 0.01
    L_bound = 1.0

    model = PINN().to(device)
    os.makedirs("results", exist_ok=True)
    w_pde, w_data, w_ic = 1.0, 50.0, 100.0

    # Fixed dataset tensors for stable L-BFGS gradient calculations
    x_p = ((2.0 * L_bound * torch.rand(2000, 1, device=device)) - L_bound).requires_grad_(True)
    y_p = ((2.0 * L_bound * torch.rand(2000, 1, device=device)) - L_bound).requires_grad_(True)
    t_p = (torch.rand(2000, 1, device=device)).requires_grad_(True)

    x_d = (2.0 * L_bound * torch.rand(1500, 1, device=device)) - L_bound
    y_d = (2.0 * L_bound * torch.rand(1500, 1, device=device)) - L_bound
    t_d = torch.rand(1500, 1, device=device)
    u_d_tgt, v_d_tgt = tg_vortex_analytic(x_d, y_d, t_d, nu=nu)

    x_i = (2.0 * L_bound * torch.rand(1500, 1, device=device)) - L_bound
    y_i = (2.0 * L_bound * torch.rand(1500, 1, device=device)) - L_bound
    t_i = torch.zeros_like(x_i).to(device)
    u_i_tgt, v_i_tgt = tg_vortex_analytic(x_i, y_i, t_i, nu=nu)

    def total_loss_func():
        loss_pde = compute_navier_stokes_loss(model, x_p, y_p, t_p, nu=nu)
        pred_d = model(torch.cat([x_d, y_d, t_d], dim=1))
        loss_data = torch.mean((u_d_tgt - pred_d[:, 0:1])**2) + torch.mean((v_d_tgt - pred_d[:, 1:2])**2)
        pred_i = model(torch.cat([x_i, y_i, t_i], dim=1))
        loss_ic = torch.mean((u_i_tgt - pred_i[:, 0:1])**2) + torch.mean((v_i_tgt - pred_i[:, 1:2])**2)
        return (w_pde * loss_pde) + (w_data * loss_data) + (w_ic * loss_ic)

    # Phase 1: Adam Optimizer (Structure Warmup)
    print("Phase 1: Running Adam for global shape convergence...")
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, 2501):
        optimizer_adam.zero_grad()
        loss = total_loss_func()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_adam.step()

    # Phase 2: L-BFGS Optimizer (High-Precision Convergence)
    print("Phase 2: Switching to L-BFGS for fine-grained error reduction...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=1000, history_size=50, tolerance_change=1e-9
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        loss = total_loss_func()
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)
    plot_verification_snapshots(model, device, nu)
