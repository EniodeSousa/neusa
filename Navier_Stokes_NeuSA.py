import torch
import torch.nn as nn

def compute_navier_stokes_loss(model, x, y, t, nu=0.01, rho=1.0):
    """
    Computes the Physics-Informed residual loss for 2D Navier-Stokes.
    x, y, t tensors must have requires_grad=True.
    """
    # Concatenate inputs: (N, 3) -> [x, y, t]
    inputs = torch.cat([x, y, t], dim=1)
    
    # Forward pass through model (Outputs: u, v, p)
    outputs = model(inputs)
    u = outputs[:, 0:1]
    v = outputs[:, 1:2]
    p = outputs[:, 2:3]

    # Compute first-order spatial and temporal gradients via automatic differentiation
    u_g = torch.autograd.grad(u, [x, y, t], torch.ones_like(u), create_graph=True)
    u_x, u_y, u_t = u_g[0], u_g[1], u_g[2]

    v_g = torch.autograd.grad(v, [x, y, t], torch.ones_like(v), create_graph=True)
    v_x, v_y, v_t = v_g[0], v_g[1], v_g[2]

    p_g = torch.autograd.grad(p, [x, y], torch.ones_like(p), create_graph=True)
    p_x, p_y = p_g[0], p_g[1]

    # Compute second-order spatial gradients
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

    # Evaluate PDE Residuals
    res_continuity = u_x + v_y
    res_momentum_u = u_t + (u * u_x) + (v * u_y) + (1.0 / rho) * p_x - nu * (u_xx + u_yy)
    res_momentum_v = v_t + (u * v_x) + (v * v_y) + (1.0 / rho) * p_y - nu * (v_xx + v_yy)

    # Compute Total Residual Mean Squared Error (MSE)
    loss_pde = (
        torch.mean(res_continuity**2) +
        torch.mean(res_momentum_u**2) +
        torch.mean(res_momentum_v**2)
    )
    
    return loss_pde, (u, v, p)
