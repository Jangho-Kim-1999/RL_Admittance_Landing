# import os
# os.environ["MUJOCO_GL"] = "egl"

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import time

# -----------------------------
# 2-link Forward Kinematics
# -----------------------------
def fk_2link(q1, q2, L1, L2):
    x = L1*np.cos(q1) + L2*np.cos(q1+q2)
    z = L1*np.sin(q1) + L2*np.sin(q1+q2)
    return np.array([x, z])

# -----------------------------
# Admittance
# -----------------------------
class Admittance:
    def __init__(self, M, B, K, Ts):
        self.M = M
        self.B = B
        self.K = K
        self.Ts = Ts

        self.z = 0.0
        self.z_old = 0.0
        self.z_old2 = 0.0
        self.Fz_old = 0.0
        self.Fz_old2 = 0.0

    def update(self, Fz):
        Ts, M, B, K = self.Ts, self.M, self.B, self.K
        c1 = 4*M + 2*B*Ts + K*(Ts**2)
        c2 = -8*M + 2*K*(Ts**2)
        c3 = 4*M - 2*B*Ts + K*(Ts**2)

        z_new = ((Ts**2)*Fz + 2*(Ts**2)*self.Fz_old + (Ts**2)*self.Fz_old2
                 - c2*self.z_old - c3*self.z_old2) / c1

        self.z_old2 = self.z_old
        self.z_old = z_new
        self.Fz_old2 = self.Fz_old
        self.Fz_old = Fz
        self.z = z_new
        return z_new

# -----------------------------
# Loading model
# -----------------------------
xml_path = "xml/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

L1, L2 = 0.25, 0.25

# Initial Condition
data.qpos[0] = 1.0
data.qpos[1] = np.pi/6
data.qpos[2] = np.pi*2/3

# Foot position reference
foot_ref = np.array([0.0, 0.25])

# Controller parameters
Kp = np.array([2000.0, 2000.0])
Kd = np.array([100.0,100.0])

# Admittance gain
M_adm, B_adm, K_adm = 5.0, 350.0, 2000.0
dt = model.opt.timestep
T_final = 2.0

adm = Admittance(M_adm, B_adm, K_adm, dt)

# -----------------------------
# Logging parameters
# -----------------------------
log = {
    "t": [], "q1": [], "q2": [], "dq1": [], "dq2": [],
    "tau1": [], "tau2": [],
    "foot_x": [], "foot_z": [], "dfoot_x": [], "dfoot_z": [],
    "Fz": [], "touch": [],
    "M": [], "B": [], "K": []
}

# -----------------------------
# Simulation
# -----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0.0, 0.0, 0.4]
    viewer.cam.distance = 2.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -10

    t = 0.0
    while t < T_final and viewer.is_running():
        q1, q2 = data.qpos[1], data.qpos[2]
        dq1, dq2 = data.qvel[1], data.qvel[2]
        dq = np.array([dq1, dq2])

        touch_val = data.sensordata[3]
        foot_pos = data.sensordata[4:7]
        foot_vel = data.sensordata[7:10]

        f_local = data.sensordata[0:3]
        theta = q1 + q2
        R = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0,             1,  0            ],
            [np.sin(theta), 0,  np.cos(theta)]
        ])
        f_world = R @ f_local
        Fz = f_world[2]

        # Admittance
        z_adm = adm.update(Fz)
        foot_ref_adm = foot_ref[1] + z_adm

        # desired foot
        foot_target = np.array([foot_ref[0], foot_ref_adm])
        foot_target_vel = np.zeros(2)

        foot_rel_pos_cal = fk_2link(q1, q2, L1, L2)
        J = np.array([
            [-L1*np.sin(q1) - L2*np.sin(q1+q2),  -L2*np.sin(q1+q2)],
            [ L1*np.cos(q1) + L2*np.cos(q1+q2),   L2*np.cos(q1+q2)]
        ])
        foot_rel_vel_cal = J @ dq

        # Task-space PD force
        F_task = Kp * (foot_target - foot_rel_pos_cal) + Kd * (foot_target_vel - foot_rel_vel_cal)

        tau = J.T @ F_task
        data.ctrl[:] = tau

        # --- Logging ---
        log["t"].append(t)
        log["q1"].append(q1)
        log["q2"].append(q2)
        log["dq1"].append(dq1)
        log["dq2"].append(dq2)
        log["tau1"].append(tau[0])
        log["tau2"].append(tau[1])
        log["foot_x"].append(foot_rel_pos_cal[0])
        log["foot_z"].append(foot_rel_pos_cal[1])
        log["dfoot_x"].append(foot_rel_vel_cal[0])
        log["dfoot_z"].append(foot_rel_vel_cal[1])
        log["Fz"].append(Fz)
        log["touch"].append(1.0 if touch_val > 1e-3 else 0.0)
        log["M"].append(adm.M)
        log["B"].append(adm.B)
        log["K"].append(adm.K)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
        t += dt

# -----------------------------
# Data Saving
# -----------------------------
df = pd.DataFrame(log)
df.to_csv("./data/conventional_1.0m.csv", index=False)
print("Saved to adm_logging.csv")
