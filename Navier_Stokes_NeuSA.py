%%writefile Navier_Stokes_NeuSA.py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 1. PINN Network Architecture
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

# 2. Physics Residual Loss
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
    return loss_pde, (u, v, p)

# 3. Snapshot Field Plotting Function
def plot_field_snapshots(model, device, save_path="results/navier_stokes_predictions.png"):
    model.eval()
    nx, ny = 100, 100
    x_lin = torch.linspace(-1, 1, nx)
    y_lin = torch.linspace(-1, 1, ny)
    grid_x, grid_y = torch.meshgrid(x_lin, y_lin, indexing="ij")
    
    time_snaps = [0.25, 0.50, 0.75, 1.00]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    with torch.no_grad():
        for i, t_val in enumerate(time_snaps):
            x_flat = grid_x.reshape(-1, 1).to(device)
            y_flat = grid_y.reshape(-1, 1).to(device)
            t_flat = torch.full_like(x_flat, t_val).to(device)

            inputs = torch.cat([x_flat, y_flat, t_flat], dim=1)
            outputs = model(inputs)
            u = outputs[:, 0:1].reshape(nx, ny).cpu().numpy()
            v = outputs[:, 1:2].reshape(nx, ny).cpu().numpy()
            p = outputs[:, 2:3].reshape(nx, ny).cpu().numpy()
            vel_mag = np.sqrt(u**2 + v**2)

            # Velocity Magnitude
            im0 = axes[0, i].imshow(vel_mag, extent=[-1, 1, -1, 1], origin="lower", cmap="viridis")
            axes[0, i].set_title(f"Vel Mag | t={t_val}s")
            fig.colorbar(im0, ax=axes[0, i])

            # Pressure Field
            im1 = axes[1, i].imshow(p, extent=[-1, 1, -1, 1], origin="lower", cmap="plasma")
            axes[1, i].set_title(f"Pressure p | t={t_val}s")
            fig.colorbar(im1, ax=axes[1, i])

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Prediction snapshot plots saved to {save_path}")

# 4. Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Starting Navier-Stokes PINN training on device: {device}")

    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    N = 1000
    x = torch.rand(N, 1, device=device, requires_grad=True)
    y = torch.rand(N, 1, device=device, requires_grad=True)
    t = torch.rand(N, 1, device=device, requires_grad=True)

    os.makedirs("results", exist_ok=True)
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        loss, _ = compute_navier_stokes_loss(model, x, y, t)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | PDE Loss: {loss.item():.6f}")

    # Save loss plot
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, label="PDE Residual Loss")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Navier-Stokes Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/navier_stokes_loss.png")
    plt.close()

    # Generate multi-time 2D spatial plots
    plot_field_snapshots(model, device)
