#!/usr/bin/env python3
import math
import torch

# ======================
# Configuración global
# ======================

device = torch.device("cpu")

T_TOTAL = 40.0

# Frecuencias
BASE_HZ  = 150.0
DT_BASE  = 1.0 / BASE_HZ

SIM_DIV  = 3     # 150/3  = 50 Hz dinámica
CTRL_DIV = 5     # 150/5  = 30 Hz control
ODOM_DIV = 30    # 150/30 = 5 Hz odometría

DT_SIM   = SIM_DIV  * DT_BASE   # 1/50
DT_CTRL  = CTRL_DIV * DT_BASE   # 1/30
DT_ODOM  = ODOM_DIV * DT_BASE   # 1/5

STEPS = int(T_TOTAL * BASE_HZ)

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
# Trayectoria de referencia (continua, evaluada en t)
# ======================

def ref_trajectory(t: torch.Tensor):
    w = OMEGA

    x = 1.5 * torch.sin(w * t)
    y = 1.5 * torch.sin(w * t) * torch.cos(w * t)
    z = 0.5 + 0.3 * torch.sin(0.5 * w * t)

    dx = 1.5 * w * torch.cos(w * t)
    dy = 1.5 * w * torch.cos(2.0 * w * t)
    dz = 0.15 * w * torch.cos(0.5 * w * t)

    yaw = torch.atan2(dy, dx)

    ddx = -1.5 * (w ** 2) * torch.sin(w * t)
    ddy = -3.0 * (w ** 2) * torch.sin(2.0 * w * t)

    denom = dx * dx + dy * dy + 1e-6
    yaw_dot = (dx * ddy - dy * ddx) / denom

    Xd = torch.stack([x, y, z, yaw])
    dXd = torch.stack([dx, dy, dz, yaw_dot])
    return Xd, dXd


# ======================
# Dinámica del dron (50 Hz)
# ======================

def dynamics_step(x, xdot, u, dt):
    """
    x:    [4] (x, y, z, yaw)
    xdot: [4] (vx, vy, vz, wyaw)
    u:    [4] (u_x, u_y, u_z, u_yaw)
    """
    yaw = x[3]
    F = rotation_F(yaw)
    xddot = F @ (Ku @ u) - (Kv @ xdot)

    xdot_new = xdot + xddot * dt
    x_new = x + xdot_new * dt
    return x_new, xdot_new


# ======================
# Medición de odometría (5 Hz, como en tu nodo)
# ======================

def compute_odom_from_state(x, xdot):
    # En tu simulador real, odometría ya es (x, xdot) sin ruido
    return x.clone(), xdot.clone()


# ======================
# Controlador inverso (30 Hz)
# ======================

def inverse_dynamic_controller(X_meas, dX_meas, Xd, dXd, gains, Ucw_prev, dt_ctrl):
    """
    gains: [12] = [Ksp(4), Ksd(4), Kp(4)]
    """
    g = gains
    Ksp = torch.diag(g[0:4])
    Ksd = torch.diag(g[4:8])
    Kp  = torch.diag(g[8:12])

    X  = X_meas
    dX = dX_meas

    # error en espacio de estados
    Xtil_raw = Xd - X
    yaw_err  = wrap_angle(Xtil_raw[3])
    Xtil = torch.stack([Xtil_raw[0], Xtil_raw[1], Xtil_raw[2], yaw_err])

    # ley en espacio "world"
    Ucw = dXd + Ksp @ torch.tanh(Kp @ Xtil)

    # derivada del comando
    dUcw = (Ucw - Ucw_prev) / dt_ctrl

    # inversión dinámica
    yaw = X[3]
    F = rotation_F(yaw)
    M = F @ Ku
    rhs = dUcw + Ksd @ (Ucw - dX) + Kv @ dX
    Udw = torch.linalg.solve(M, rhs)

    # empaque a formato 6D (tipo cmd_vel extendido)
    Ud = torch.zeros(6, dtype=torch.float32, device=device)
    Ud[0:3] = Udw[0:3]
    Ud[5]   = Udw[3]

    return Ud, Ucw


# ======================
# Simulación + Costo (prioriza tracking)
# ======================

def simulate_episode(gains: torch.Tensor) -> torch.Tensor:

    x    = torch.zeros(4, device=device)  # [x, y, z, yaw]
    xdot = torch.zeros(4, device=device)  # [vx, vy, vz, wyaw]

    # odometría inicial
    X_meas, dX_meas = compute_odom_from_state(x, xdot)

    Ucw_prev = torch.zeros(4, device=device)
    u_cmd    = torch.zeros(6, device=device)  # último comando de control aplicado

    cost = torch.tensor(0.0, device=device)
    n_odom_samples = 0

    # pesos de costo (priorizar tracking)
    w_track_pos = 1.0
    w_track_yaw = 0.5
    w_vel       = 0.05
    w_u         = 0.002
    w_acc       = 0.002

    for k in range(STEPS):
        t = torch.tensor(k * DT_BASE, dtype=torch.float32, device=device)

        # referencia continua en tiempo t
        Xd, dXd = ref_trajectory(t)

        # --- ODOMETRÍA + COSTO (5 Hz) ---
        if k % ODOM_DIV == 0:
            X_meas, dX_meas = compute_odom_from_state(x, xdot)

            pos_err = Xd[0:3] - X_meas[0:3]
            yaw_err = wrap_angle(Xd[3] - X_meas[3])

            loss_track = w_track_pos * (pos_err**2).sum() + \
                         w_track_yaw * (yaw_err**2)

            loss_vel = w_vel * (dX_meas[0:3]**2).sum()

            cost = cost + (loss_track + loss_vel)
            n_odom_samples += 1

        # --- CONTROL (30 Hz) ---
        if k % CTRL_DIV == 0:
            Ud, Ucw_prev = inverse_dynamic_controller(
                X_meas, dX_meas, Xd, dXd, gains, Ucw_prev, DT_CTRL
            )

            Ud_clipped = torch.clamp(Ud, -U_SAT, U_SAT)
            u_cmd = Ud_clipped

            # penalizar energía de control (suave)
            cost = cost + w_u * (Ud_clipped[0:3]**2).sum()

        # --- DINÁMICA (50 Hz) ---
        if k % SIM_DIV == 0:
            # mando que llega al modelo
            u_dyn = torch.stack([
                u_cmd[0],
                u_cmd[1],
                u_cmd[2],
                u_cmd[5]
            ])

            # penalizar aceleraciones fuertes (aprox)
            yaw = x[3]
            F = rotation_F(yaw)
            xddot_proxy = F @ (Ku @ u_dyn) - Kv @ xdot
            cost = cost + w_acc * (xddot_proxy[0:3]**2).sum()

            # integrar dinámica
            x, xdot = dynamics_step(x, xdot, u_dyn, DT_SIM)

    if n_odom_samples > 0:
        cost = cost / n_odom_samples

    return cost


# ======================
# Optimización con Adam
# ======================

def main():

    init_gains = torch.tensor([
        0.2948657274246216,
        0.282520592212677,
        0.33964332938194275,
        2.4078164100646973,
        4.946599960327148,
        4.323536396026611,
        6.013118267059326,
        10.796932220458984,
        0.29646456241607666,
        0.28467491269111633,
        3.555035352706909,
        1.9480745792388916
    ], dtype=torch.float32, device=device)

    gains = torch.nn.Parameter(init_gains.clone())
    optimizer = torch.optim.Adam([gains], lr=1e-1)

    n_epochs = 200

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        cost = simulate_episode(gains)
        cost.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("Epoch =", epoch+1)
            print("Cost  =", float(cost))
            print("Gains =", gains.detach().cpu().tolist())
            print("-" * 40)

    print("# --- Optimización terminada ---")
    print("Gains_opt =", gains.detach().cpu().tolist())


if __name__ == "__main__":
    main()
