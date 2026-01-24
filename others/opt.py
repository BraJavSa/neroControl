#!/usr/bin/env python3
import numpy as np
from math import sin, cos, pi, atan2
from cmaes import CMA

# ===================================
#  Modelo dinámico simplificado Bebop
# ===================================
Model_simp = np.array([
    0.8417, 0.18227,
    0.8354, 0.17095,
    3.966,  4.001,
    9.8524, 4.7295
])

Ku = np.diag([Model_simp[0], Model_simp[2], Model_simp[4], Model_simp[6]])
Kv = np.diag([Model_simp[1], Model_simp[3], Model_simp[5], Model_simp[7]])

# ==============================
#  Trayectoria tipo ∞ (offline)
# ==============================
dt = 1.0/30.0
T = 40.0
N = int(T/dt)
omega = 2*pi/T

def generate_ref():
    ref = np.zeros((N,7))
    for k in range(N):
        t = k*dt
        x = 1.5*np.sin(omega*t)
        y = 1.5*np.sin(omega*t)*np.cos(omega*t)
        z = 1.0 + 0.3*np.sin(0.5*omega*t) - 0.5

        # derivadas finitas
        t_prev = t - dt
        xp = 1.5*np.sin(omega*t_prev)
        yp = 1.5*np.sin(omega*t_prev)*np.cos(omega*t_prev)
        zp = 1.0 + 0.3*np.sin(0.5*omega*t_prev) - 0.5

        dx = (x - xp)/dt
        dy = (y - yp)/dt
        dz = (z - zp)/dt

        yaw = atan2(dy, dx)
        ref[k,:] = [x,y,z,yaw,dx,dy,dz]

    return ref

ref = generate_ref()

# =============================
#     Simulación + Control
# =============================
uSat = np.array([1,1,1,1])

def simulate_cost(g):
    g = np.array(g)
    Ksp = np.diag(g[0:4])
    Ksd = np.diag(g[4:8])
    Kp  = np.diag(g[8:12])

    x = np.zeros(4)    # [x,y,z,yaw]
    xdot = np.zeros(4)
    cost = 0.0

    for k in range(N):
        xd,yd,zd,psid,dxd,dyd,dzd = ref[k]
        X  = np.array([x[0],x[1],x[2],x[3]])
        dX = np.array([xdot[0],xdot[1],xdot[2],xdot[3]])
        Xd = np.array([xd,yd,zd,psid])
        dXd= np.array([dxd,dyd,dzd,0.0])

        # error de orientación
        Xtil = Xd - X
        if abs(Xtil[3]) > pi:
            Xtil[3] -= 2*pi*np.sign(Xtil[3])

        # controlador cinemático + compensador
        Ucw = dXd + Ksp @ np.tanh(Kp @ Xtil)
        Udw = Ksd @ (Ucw - dX)

        psi = x[3]
        F = np.array([
            [cos(psi),-sin(psi),0,0],
            [sin(psi), cos(psi),0,0],
            [0,0,1,0],
            [0,0,0,1]
        ])

        # inversión aproximada
        u = np.linalg.pinv(Ku) @ np.linalg.pinv(F) @ (Udw + Kv@dX)
        u = np.clip(u, -uSat, uSat)

        # dinamica
        xddot = F@(Ku@u) - Kv@xdot
        xdot += xddot*dt
        x += xdot*dt

        cost += np.sum((Xd - X)**2)

    return cost/N


# =============================
#       CMA-ES (cmaes)
# =============================
x0 = np.array([
    1.1,1.1,3.0,1.5,
    0.8,0.8,1.8,1.2,
    1.6,1.6,1.0,1.5
])

lb = np.full(12, 0.1)
ub = np.full(12, 5.0)
bounds = np.column_stack((lb, ub))  # (12×2)

optimizer = CMA(
    mean=x0,
    sigma=0.1,
    bounds=bounds,
    population_size=18
)

maxiter = 60
best = (x0, float("inf"))

for gen in range(maxiter):
    solutions=[]
    for _ in range(optimizer.population_size):
        x = optimizer.ask()
        f = simulate_cost(x)
        solutions.append((x,f))
        if f < best[1]:
            best = (x,f)

    optimizer.tell(solutions)
    print(f"[GEN {gen:02d}] best cost = {best[1]:.5f}")

print("\n===================")
print(" BEST GAINS FOUND ")
print("===================")
print(best[0])
