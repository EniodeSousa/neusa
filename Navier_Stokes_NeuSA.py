%%writefile Navier_Stokes_NeuSA.py
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
import torch.nn as nn
from IPython.display import Image, display

# 1. PINN Network Architecture (unchanged)
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

# 2. Physics Residual Loss Function (unchanged)
def compute_navier_stokes_loss(model, x, y, t, nu=0.01, rho=1.0):
    inputs = torch.cat([x, y, t], dim=1)
    outputs = model(inputs)
    u, v, p = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]

    u_g = torch.autograd.grad(u, [x, y, t], torch.ones_like(u), create_graph=True)
    u_x, u_y, u_t = u_g[0], u_g[1], u_g[2]

    v_g = torch.autograd.grad(v, [x, y, t], torch.ones_like(v), create_graph=True)
    v_x, v_y, v_t = v_g[0], v_g[1], v_g[2]

    p_g = torch.autograd.grad(p, [x, y], torch.ones_like(p), create_graph=True)
    p_x, p_y = p_g[0], p_g[1]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

    res_continuity = u_x + v_y
    res_momentum_u = u_t + (u * u_x) + (v * u_y) + (1.0 / rho) * p_x - nu * (u_xx + u_yy)
    res_momentum_v = v_t + (u * v_x) + (v * v_y) + (1.0 / rho) * p_y - nu * (v_xx + v_yy)

    loss_pde = (
        torch.mean(res_continuity**2) +
        torch.mean(res_momentum_u**2) +
        torch.mean(res_momentum_v**2)
    )
    return loss_pde

# 3. Taylor-Green Vortex Analytical Solution (FIXED t handling)
def tg_vortex_analytic(x, y, t, nu=0.01):
    # k = pi maps spatial period to [-1, 1]
    
    # --- FIX START ---
    # We must ensure 't' is a tensor before passed to torch.exp()
    # It might be passed as a float from the plotting loop[cite: 5, 8].
    if not torch.is_tensor(t):
        # Cast to tensor, match device of x, and enforce float32/float64
        t = torch.tensor(t, device=x.device, dtype=x.dtype)
    # --- FIX END ---
        
    u = -torch.cos(torch.pi * x) * torch.sin(torch.pi * y) * torch.exp(-2.0 * nu * (torch.pi**2) * t)
    v = torch.sin(torch.pi * x) * torch.cos(torch.pi * y) * torch.exp(-2.0 * nu * (torch.pi**2) * t)
    p = -0.25 * (torch.cos(2.0 * torch.pi * x) + torch.cos(2.0 * torch.pi * y)) * torch.exp(-4.0 * nu * (torch.pi**2) * t)
    return u, v, p

# 4. Amplitudes Snapshot Plotting Routine (unchanged)
def plot_verification_snapshots(model, device, nu, save_path="results/navier_stokes_verification.png"):
    model.eval()
    nx, ny = 100, 100
    L = 1.0 # domain [-L, L]
    x_lin = torch.linspace(-L, L, nx)
    y_lin = torch.linspace(-L, L, ny)
    grid_x, grid_y = torch.meshgrid(x_lin, y_lin, indexing="ij")
    
    time_snaps = [0.25, 0.50, 0.75, 1.00]
    # Rows: u_tgt, v_tgt, u_pred, v_pred, u_err, v_err (6 rows)
    fig, axes = plt.subplots(6, 4, figsize=(16, 18))

    with torch.no_grad():
        for i, t_val in enumerate(time_snaps):
            x_flat = grid_x.reshape(-1, 1).to(device)
            y_flat = grid_y.reshape(-1, 1).to(device)
            # t_flat is a tensor:
            t_flat = torch.full_like(x_flat, t_val).to(device)
            inputs = torch.cat([x_flat, y_flat, t_flat], dim=1)

            # --- Traceback line 84: tg_vortex_analytic receives t_val (float) ---
            # Inside the loop, t_val is a float. tg_vortex_analytic now handles it[cite: 5, 8].
            u_tgt, v_tgt, p_tgt = tg_vortex_analytic(x_flat, y_flat, t_val, nu)
            U_tgt = u_tgt.reshape(nx, ny).cpu().numpy()
            V_tgt = v_tgt.reshape(nx, ny).cpu().numpy()
            
            # Get Predictions
            outputs = model(inputs)
            U_pred = outputs[:, 0:1].reshape(nx, ny).cpu().numpy()
            V_pred = outputs[:, 1:2].reshape(nx, ny).cpu().numpy()

            # Absolute Errors
            U_err = np.abs(U_tgt - U_pred)
            V_err = np.abs(V_tgt - V_pred)

            # --- ROW 0: Target U ---
            im = axes[0, i].imshow(U_tgt, extent=[-L, L, -L, L], origin="lower", cmap="bwr", vmin=-1.0, vmax=1.0)
            axes[0, i].set_title(f"Target u (TGV) | t={t_val}s")
            fig.colorbar(im, ax=axes[0, i])
            # --- ROW 1: Target V ---
            im = axes[1, i].imshow(V_tgt, extent=[-L, L, -L, L], origin="lower", cmap="bwr", vmin=-1.0, vmax=1.0)
            axes[1, i].set_title(f"Target v (TGV) | t={t_val}s")
            fig.colorbar(im, ax=axes[1, i])
            
            # --- ROW 2: Predicted U ---
            im = axes[2, i].imshow(U_pred, extent=[-L, L, -L, L], origin="lower", cmap="bwr", vmin=-1.0, vmax=1.0)
            axes[2, i].set_title(f"Predicted u | t={t_val}s")
            fig.colorbar(im, ax=axes[2, i])
            # --- ROW 3: Predicted V ---
            im = axes[3, i].imshow(V_pred, extent=[-L, L, -L, L], origin="lower", cmap="bwr", vmin=-1.0, vmax=1.0)
            axes[3, i].set_title(f"Predicted v | t={t_val}s")
            fig.colorbar(im, ax=axes[3, i])
            
            # --- ROW 4: U Error (Log Scale) ---
            im = axes[4, i].imshow(U_err, extent=[-L, L, -L, L], origin="lower", cmap="plasma", norm=LogNorm(vmin=1e-5, vmax=1e-1))
            axes[4, i].set_title(f"Abs Error |u_tgt - u_pred|")
            fig.colorbar(im, ax=axes[4, i])
            # --- ROW 5: V Error (Log Scale) ---
            im = axes[5, i].imshow(V_err, extent=[-L, L, -L, L], origin="lower", cmap="plasma", norm=LogNorm(vmin=1e-5, vmax=1e-1))
            axes[5, i].set_title(f"Abs Error |v_tgt - v_pred|")
            fig.colorbar(im, ax=axes[5, i])

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Verified verification comparison plots saved to {save_path}")

# 5. Main Execution Block
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device)
    nu = 0.01 # Viscount benchmark kinematic viscosity
    print(f"Starting TGV Verification on device: {device}. Viscosity nu={nu}.")

    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("results", exist_ok=True)
    loss_history = []
    L_bound = 1.0 # domain [-1, 1]
    N_pde = 1000 # collocation points

    print("Beginning unbuffered training loop (PDE Loss real-time streaming)...")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        
        # Sample collocation points: (x, y) ~ Unif(-1, 1), t ~ Unif(0, 1)
        x = (2.0 * L_bound * torch.rand(N_pde, 1, device=device)) - L_bound
        y = (2.0 * L_bound * torch.rand(N_pde, 1, device=device)) - L_bound
        t = torch.rand(N_pde, 1, device=device) # t=[0, 1]
        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True
        
        loss = compute_navier_stokes_loss(model, x, y, t, nu=nu)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if epoch % 100 == 0 or epoch == 1:
            # We must use -u in Colab to stream these logs line-by-line[cite: 1, 11]
            print(f"Epoch {epoch:4d}/{args.epochs} | PDE Loss: {loss.item():.6f}")

    # Plot loss history
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, label="Combined PINN Loss")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("TGV Verification Training Progress")
    plt.legend(); plt.grid(True)
    plt.savefig("results/tgv_verification_loss.png")
    plt.close()

    # --- Line 179 traceback: plot routine called ---
    plot_verification_snapshots(model, device, nu)
