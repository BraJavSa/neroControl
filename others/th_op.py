#!/usr/bin/env python3
import math
import torch

# ======================
# Configuración global
# ======================

device = torch.device("cpu")
DT = 1.0 / 30.0
T_TOTAL = 40.0
OMEGA = 2.0 * math.pi / T_TOTAL
U_SAT = 1.0  # saturación de comandos

# ======================
# Parámetros del modelo (como en tu código)
# ======================

MODEL_SIMP = torch.tensor([
    0.8417, 0.18227,
    0.8354, 0.17095,
    3.966,  4.001,
    9.8524, 4.7295
], dtype=torch.float32, device=device)

Ku_diag = MODEL_SIMP[[0, 2, 4, 6]]  # [0, 2, 4, 6]
Kv_diag = MODEL_SIMP[[1, 3, 5, 7]]  # [1, 3, 5, 7]

Ku = torch.diag(Ku_diag)
Kv = torch.diag(Kv_diag)


# ======================
# Utilidades
# ======================

def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def rotation_F(yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    F = torch.zeros(4, 4, dtype=torch.float32, device=device)
    F[0, 0] = c
    F[0, 1] = -s
    F[1, 0] = s
    F[1, 1] = c
    F[2, 2] = 1.0
    F[3, 3] = 1.0
    return F


# ======================
# NUEVA referencia: poses del cubo (30 Hz)
# ======================

def ref_cube(t: torch.Tensor):
    L = 1.5
    points = torch.tensor([
        [0.0,   0.0,   1.5],
        [ L/2,  L/2,   1.7],
        [-L/2,  L/2,   1.4],
        [-L/2, -L/2,   1.8],
        [ L/2, -L/2,   1.2],
    ], device=device, dtype=torch.float32)

    yaws = torch.deg2rad(torch.tensor([0, 50, 150, 180, 210],
                                      device=device, dtype=torch.float32))

    hold_time = 12.0
    idx = torch.clamp((t / hold_time).long(), 0, len(points)-1)

    pos = points[idx]
    yaw = yaws[idx]

    Xd = torch.stack([pos[0], pos[1], pos[2], yaw])
    dXd = torch.zeros_like(Xd)  # velocidades 0
    return Xd, dXd


# ======================
# Dinámica del dron
# ======================

def dynamics_step(x: torch.Tensor,
                  xdot: torch.Tensor,
                  u: torch.Tensor,
                  dt: float):
    yaw = x[3]
    F = rotation_F(yaw)
    xddot = F @ (Ku @ u) - (Kv @ xdot)

    xdot_new = xdot + xddot * dt
    x_new = x + xdot_new * dt
    return x_new, xdot_new


# ======================
# Medición de odometría (5 Hz)
# ======================

def compute_odom_from_state(x: torch.Tensor,
                            xdot: torch.Tensor):
    X_meas = x.clone()
    dX_meas = xdot.clone()
    return X_meas, dX_meas


# ======================
# Controlador inverso (Torch)
# ======================

def inverse_dynamic_controller(X_meas, dX_meas, Xd, dXd, gains, Ucw_prev, dt):

    g = gains
    Ksp = torch.diag(g[0:4])
    Ksd = torch.diag(g[4:8])
    Kp  = torch.diag(g[8:12])

    X = X_meas
    dX = dX_meas

    Xtil_raw = Xd - X
    yaw_err = wrap_angle(Xtil_raw[3])
    Xtil = torch.stack([Xtil_raw[0], Xtil_raw[1], Xtil_raw[2], yaw_err])

    Ucw = dXd + Ksp @ torch.tanh(Kp @ Xtil)
    dUcw = (Ucw - Ucw_prev) / dt

    yaw = X[3]
    F = rotation_F(yaw)
    M = F @ Ku
    rhs = dUcw + Ksd @ (Ucw - dX) + Kv @ dX
    Udw = torch.linalg.solve(M, rhs)

    Ud = torch.zeros(6, dtype=torch.float32, device=device)
    Ud[0:3] = Udw[0:3]
    Ud[5] = Udw[3]

    return Ud, Ucw


# ======================
# Nueva simulación + costo
# ======================

def simulate_episode(gains: torch.Tensor) -> torch.Tensor:

    steps = int(T_TOTAL / DT)
    odom_period = 6  # 30 Hz / 5 Hz

    x = torch.zeros(4, dtype=torch.float32, device=device)
    xdot = torch.zeros(4, dtype=torch.float32, device=device)

    X_meas, dX_meas = compute_odom_from_state(x, xdot)
    Ucw_prev = torch.zeros(4, dtype=torch.float32, device=device)

    cost = torch.tensor(0.0, dtype=torch.float32, device=device)
    n_odom_samples = 0

    for k in range(steps):
        t = torch.tensor(k * DT, dtype=torch.float32, device=device)

        Xd, dXd = ref_cube(t)

        if k % odom_period == 0:
            X_meas, dX_meas = compute_odom_from_state(x, xdot)

            pos_err = Xd[0:3] - X_meas[0:3]
            yaw_err = wrap_angle(Xd[3] - X_meas[3])

            loss_track = (pos_err**2).sum() + 0.5*(yaw_err**2)
            loss_vel = 0.1 * (dX_meas[0:3]**2).sum()

            loss = loss_track + loss_vel
            cost = cost + loss
            n_odom_samples += 1

        Ud, Ucw_prev = inverse_dynamic_controller(
            X_meas, dX_meas, Xd, dXd, gains, Ucw_prev, DT
        )

        Ud_clipped = torch.clamp(Ud, -U_SAT, U_SAT)
        cost = cost + 0.01*(Ud_clipped[0:3]**2).sum()

        u_dyn = torch.stack([
            Ud_clipped[0],
            Ud_clipped[1],
            Ud_clipped[2],
            Ud_clipped[5]
        ])

        x, xdot = dynamics_step(x, xdot, u_dyn, DT)

    if n_odom_samples > 0:
        cost = cost / n_odom_samples

    return cost


# ======================
# Optimización con Adam
# ======================

def main():

    init_gains = torch.tensor([1.2806, 1.2804, 3.2648, 1.7506, 0.9505, 0.9507, 2.0379, 1.4151, 1.8123, 1.8121, 1.1834, 1.7449], dtype=torch.float32, device=device)

    gains = torch.nn.Parameter(init_gains.clone())
    optimizer = torch.optim.Adam([gains], lr=1e-2)

    n_epochs = 200

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        cost = simulate_episode(gains)
        cost.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("Epoch =", epoch+1)
            print("Cost =", float(cost))
            print("Gains =", gains.detach().cpu().tolist())

    print("# --- Optimización terminada ---")
    print("Gains_opt =", gains.detach().cpu().tolist())


if __name__ == "__main__":
    main()
