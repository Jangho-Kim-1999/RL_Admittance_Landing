#!/usr/bin/env python
import os
import sys
import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from proposed_train import OneLegLandingEnv


def map_action_to_gains(a, M_min, M_max, B_min, B_max, K_min, K_max):
    a = np.clip(a, -1.0, 1.0)
    M = M_min + (a[0] + 1.0) * 0.5 * (M_max - M_min)
    B = B_min + (a[1] + 1.0) * 0.5 * (B_max - B_min)
    K = K_min + (a[2] + 1.0) * 0.5 * (K_max - K_min)
    return float(np.clip(M, M_min, M_max)), \
           float(np.clip(B, B_min, B_max)), \
           float(np.clip(K, K_min, K_max))


def test_trained_model(model_path, xml_path="xml/scene.xml", sim_time=2.0):
    env = OneLegLandingEnv(xml_path=xml_path, render_mode="human")
    obs, _ = env.reset()
    model = PPO.load(model_path)

    print("=== Start Test ===")

    M_min, M_max = 1.0, 10.0
    B_min, B_max = 0.0, 500.0
    K_min, K_max = 0.0, 2000.0

    Ms, Bs, Ks, timesteps, Fzs = [], [], [], [], []

    max_steps = int(sim_time / (env.dt * env.frame_skip))

    # --------- Plot setup ---------
    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(8, 6))

    # (1) M
    line_M, = axs[0].plot([], [], color="blue", label="M (kg)")
    axs[0].set_ylabel("M (kg)")
    axs[0].legend(); axs[0].grid()

    # (2) B
    line_B, = axs[1].plot([], [], color="green", label="B (Ns/m)")
    axs[1].set_ylabel("B (Ns/m)")
    axs[1].legend(); axs[1].grid()

    # (3) K
    line_K, = axs[2].plot([], [], color="red", label="K (N/m)")
    axs[2].set_ylabel("K (N/m)")
    axs[2].legend(); axs[2].grid()

    # (4) Force
    line_Fz, = axs[3].plot([], [], color="purple",label="Fz (N)")
    axs[3].set_xlabel("Step")
    axs[3].set_ylabel("Force [N]")
    axs[3].legend(); axs[3].grid()

    plt.tight_layout()
    plt.show(block=False)

    # --------- Simulation ---------
    ep_return, ep_len = 0.0, 0
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        M, B, K = map_action_to_gains(action, M_min, M_max, B_min, B_max, K_min, K_max)

        Fz_norm = obs[8]
        Fz = Fz_norm * 100.0

        Ms.append(M); Bs.append(B); Ks.append(K); Fzs.append(Fz)
        timesteps.append(step)

        # Real time plotting

        line_M.set_data(timesteps, Ms)
        axs[0].relim(); axs[0].autoscale_view()

        line_B.set_data(timesteps, Bs)
        axs[1].relim(); axs[1].autoscale_view()

        line_K.set_data(timesteps, Ks)
        axs[2].relim(); axs[2].autoscale_view()

        line_Fz.set_data(timesteps, Fzs)
        axs[3].relim(); axs[3].autoscale_view()

        plt.pause(0.001)

        time.sleep(0.01)

        ep_return += reward
        ep_len += 1

    print(f"Episode finished. return={ep_return:.2f}, length={ep_len}")
    env.close()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "proposed_20000000_steps.zip"

    model_path = os.path.join("", "./models", model_name)
    test_trained_model(model_path, sim_time=5.0)
