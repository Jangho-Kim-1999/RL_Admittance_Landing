import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


# --------------------- Utility --------------------
def fk_2link(q1, q2, L1, L2):
    x = L1*np.cos(q1) + L2*np.cos(q1+q2)
    z = L1*np.sin(q1) + L2*np.sin(q1+q2)
    return np.array([x, z])

def jacb(q1, q2, L1, L2):
    J = np.array([
        [-L1*np.sin(q1) - L2*np.sin(q1+q2),   -L2*np.sin(q1+q2)],
        [ L1*np.cos(q1) + L2*np.cos(q1+q2),    L2*np.cos(q1+q2)]
    ])
    return J


# --------------------- Admittance ---------------------
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

    def reset(self, z0=0.0):
        self.z = z0
        self.z_old = z0
        self.z_old2 = z0
        self.Fz_old = 0.0
        self.Fz_old2 = 0.0

    def update(self, Fz):
        Ts, M, B, K = self.Ts, self.M, self.B, self.K
        c1 = 4*M + 2*B*Ts + K*(Ts**2)
        c2 = -8*M + 2*K*(Ts**2)
        c3 = 4*M - 2*B*Ts + K*(Ts**2)
        z_new = ( (Ts**2)*Fz + 2*(Ts**2)*self.Fz_old + (Ts**2)*self.Fz_old2
                  - c2*self.z_old - c3*self.z_old2 ) / c1
        self.z_old2 = self.z_old
        self.z_old = z_new
        self.Fz_old2 = self.Fz_old
        self.Fz_old = Fz
        self.z = z_new
        return z_new


# --------------------- MuJoCo Env ---------------------
class OneLegLandingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 200}

    def __init__(self, xml_path, render_mode="human", frame_skip=1,

                 #! Iniitial condition
                 q_initial=(math.pi/6, math.pi*2/3),
                 L1=0.25, L2=0.25,
                 foot_ref=(0.0, 0.25),
                 kp=(2000.0, 2000.0), kd=(100, 100)):

        super().__init__()
        assert os.path.exists(xml_path), f"XML not found: {xml_path}"

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.frame_skip  = frame_skip
        self.dt          = self.model.opt.timestep
        self.t           = 0

        #! Max Steps
        self.max_steps   = 2000

        # link lengths
        self.L1, self.L2 = float(L1), float(L2)
        self.Kp = np.array(kp, dtype=np.float32)
        self.Kd = np.array(kd, dtype=np.float32)
        self.u_max = np.array([60.0, 60.0], dtype=np.float32)

        self.q1_initial, self.q2_initial = float(q_initial[0]), float(q_initial[1])
        self.foot_ref = np.array([foot_ref[0], foot_ref[1]], dtype=np.float32)

        #! Gain range
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.M_min, self.M_max = 1, 10.0
        self.B_min, self.B_max = 0.0, 500.0
        self.K_min, self.K_max = 0.0, 2000.0

        #! observation space
        high = np.array([np.inf]*10, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.admittance = None
        self.last_tau = np.zeros(2, dtype=np.float32)
        self.last_gains = np.array([2.0, 50.0, 1000.0], dtype=np.float32)

        # --- Force peak tracking ---
        self.Fz_ema = 0.0
        self.Fz_peak = 0.0
        self.alpha_ema = 1.0


    def _map_action_to_gains(self, a):
        a = np.clip(a, -1.0, 1.0)

        M = self.M_min + (a[0] + 1.0) * 0.5 * (self.M_max - self.M_min)
        B = self.B_min + (a[1] + 1.0) * 0.5 * (self.B_max - self.B_min)
        K = self.K_min + (a[2] + 1.0) * 0.5 * (self.K_max - self.K_min)

        return float(M), float(B), float(K)

    def _sanitize_obs(self, obs):
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    def _get_force_world(self,theta):
        f_local = self.data.sensordata[0:3]
        R = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0,             1,  0            ],
            [np.sin(theta), 0,  np.cos(theta)]
        ], dtype=np.float64)
        return R @ f_local

    def get_obs(self):

        q1, q2 = self.data.qpos[1], self.data.qpos[2]
        dq1, dq2 = self.data.qvel[1], self.data.qvel[2]
        raw_touch = self.data.sensordata[3]

        foot_rel_pos = fk_2link(q1, q2, self.L1, self.L2)
        J = jacb(q1, q2, self.L1, self.L2)
        dq = np.array([dq1, dq2])
        foot_rel_vel = J @ dq

        touch_flag = 1.0 if raw_touch > 1e-3 else 0.0
        Fz = float(self._get_force_world(q1+q2)[2])

        obs = np.array([
            q1/3.14, q2/3.14,
            dq1/31.4, dq2/31.4,
            foot_rel_pos[0]/0.5, foot_rel_pos[1]/0.5,
            foot_rel_vel[0]/5, foot_rel_vel[1]/5,
            Fz/100.0,
            touch_flag
        ], dtype=np.float32)

        return self._sanitize_obs(obs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.t = 0
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        #! Train
        # self.data.qpos[0] = np.random.uniform(0.3, 1.0)

        #! Test
        self.data.qpos[0] = 0.3

        self.data.qpos[1] = self.q1_initial
        self.data.qpos[2] = self.q2_initial
        mujoco.mj_forward(self.model, self.data)

        #! Initialization
        self.admittance = Admittance(5.0, 50.0, 1000.0, self.dt)
        self.admittance.reset(0.0)
        self.last_tau[:] = 0.0
        self.success_counter = 0  # ← 추가
        self.Fz_ema = 0.0
        self.Fz_peak = 0.0


        return self.get_obs(), {}

    def step(self, action):

        # 1) Action → M,B,K
        M_adm, B_adm, K_adm = self._map_action_to_gains(action)
        self.admittance.M, self.admittance.B, self.admittance.K = M_adm, B_adm, K_adm
        gains_now = np.array([M_adm, B_adm, K_adm], dtype=np.float32)

        # 2) Sensing
        q = self.data.qpos[1:3].copy()
        dq = self.data.qvel[1:3].copy()
        Fz = float(self._get_force_world(q[0]+q[1])[2])

        #! 3) Admittance Controller
        z_adm = self.admittance.update(Fz)
        foot_des_x = self.foot_ref[0]
        foot_des_z = self.foot_ref[1] + z_adm

        foot_target = np.array([foot_des_x, foot_des_z])
        foot_target_vel = np.zeros(2)

        foot_rel_pos = fk_2link(q[0], q[1], self.L1, self.L2)
        J = jacb(q[0], q[1], self.L1, self.L2)
        dq = np.array([dq[0], dq[1]])
        foot_rel_vel = J @ dq

        # 4) Task-space PD force
        F_task = self.Kp * (foot_target - foot_rel_pos) + self.Kd * (foot_target_vel - foot_rel_vel)

        # 5) Joint torques
        tau = J.T @ F_task
        tau = np.clip(tau, -self.u_max, +self.u_max)
        self.data.ctrl[:] = tau

        self.last_tau = tau.astype(np.float32)

        # 6) Simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.t += 1

        # 7) Observation
        obs = self.get_obs()
        (
            q1_n, q2_n,
            dq1_n, dq2_n,
            foot_x_n, foot_z_n,
            dfoot_x_n, dfoot_z_n,
            Fz_norm,
            touch_flag
        ) = obs

        # Scaling down
        q1, q2 = q1_n * 3.14, q2_n * 3.14
        dq1, dq2 = dq1_n * 31.4, dq2_n * 31.4
        foot_x_rel, foot_z_rel = foot_x_n * 0.5, foot_z_n * 0.5
        dfoot_x_rel, dfoot_z_rel = dfoot_x_n * 5.0, dfoot_z_n * 5.0
        Fz = Fz_norm * 100.0   # 정규화 풀기
        W = 3.02 * 9.81

        #! 8) Reward function design
        # (1) foot position
        x_ref, z_ref = 0.0, 0.23
        r_foot = -(((foot_x_rel - x_ref)/0.5)**2 + ((foot_z_rel - z_ref)/0.5)**2)

        # (2) grf minimization
        r_force = -((-Fz - W)/W)**2 if touch_flag > 0.5 else 0.0
        r_peak = 0.0
        if touch_flag > 0.5:
            Fz_abs = abs(Fz)
            self.Fz_ema = self.alpha_ema * Fz_abs + (1 - self.alpha_ema) * self.Fz_ema
            if self.Fz_ema > self.Fz_peak:
                r_peak = -((self.Fz_ema - self.Fz_peak) / W)
                self.Fz_peak = self.Fz_ema

        # (3) joint angular velocity / foot velocity minimization
        r_dq = -((dq1/31.4)**2 + (dq2/31.4)**2)
        r_dfoot = -((dfoot_x_rel/5)**2 + (dfoot_z_rel/5)**2)

        # (4) torque mionimization
        tau1_n = self.last_tau[0] / self.u_max[0]
        tau2_n = self.last_tau[1] / self.u_max[1]
        r_tau = -(tau1_n**2 + tau2_n**2)

        # (5) alive bonus
        alive_b = 5.0

        # weight
        w_foot   = 2.0
        w_force  = 1.0
        w_peak   = 1.0
        w_dq     = 0.05
        w_dfoot  = 0.05
        w_tau    = 0.02

        # total reward
        reward = (
            w_foot  * r_foot +
            w_force * r_force +
            w_peak  * r_peak +
            w_dq    * r_dq +
            w_dfoot * r_dfoot +
            w_tau   * r_tau +
            alive_b
        )

        terminated = bool((foot_z_rel < -0.1) or (foot_z_rel > 0.5))
        truncated  = bool(self.t >= self.max_steps)
        if terminated:
            reward -= 20.0

        vel_tol = 0.01
        success_cond = (
            touch_flag > 0.5 and
            abs(foot_x_rel - x_ref) < 0.02 and
            abs(foot_z_rel - z_ref) < 0.02 and
            abs(dfoot_x_rel) < vel_tol and
            abs(dfoot_z_rel) < vel_tol
        )

        if success_cond:
            self.success_counter += 1
        else:
            self.success_counter = 0

        success_return = 0.0
        N = 50
        if self.success_counter >= N:
            success_return = 200.0
            reward += success_return

        self.last_gains = gains_now
        return obs, float(reward), terminated, truncated, {}

# Rendering
    def render(self):
        if self.render_mode != "human": return
        if not hasattr(self, "_viewer"):
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

            cam_follow = self.data.qpos[0]
            self._viewer.cam.lookat[:] = [0.2, 0.0, 0.15]
            self._viewer.cam.distance = 1.0
            self._viewer.cam.azimuth = 90
            self._viewer.cam.elevation = 0

        self._viewer.sync()

    def close(self):
        if hasattr(self, "_viewer"):
            self._viewer.close()


# ------------------------------- main -------------------------------
if __name__ == "__main__":
    XML_PATH = "xml/scene.xml"
    env = OneLegLandingEnv(xml_path=XML_PATH, render_mode="human")
    model = PPO(
        policy="MlpPolicy", env=env, device="cpu", verbose=1,
        learning_rate=3e-4, n_steps=2048, batch_size=256,
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=1e-3, vf_coef=0.5, seed=42
    )

    #! Training setup
    checkpoint_callback = CheckpointCallback(
        save_freq=2048, save_path="./models/", name_prefix="proposed"
    )
    model.learn(total_timesteps=50_000_000, callback=checkpoint_callback)
    model.save("proposed")
    print("Model saved as proposed.zip")
