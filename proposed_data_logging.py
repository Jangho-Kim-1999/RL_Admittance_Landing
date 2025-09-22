#!/usr/bin/env python
import os
import sys
import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import pandas as pd   # CSV 저장에 사용
from stable_baselines3 import PPO
from proposed_train import OneLegLandingEnv  # 학습 환경 import


def map_action_to_gains(a, M_min, M_max, B_min, B_max, K_min, K_max):
    a = np.clip(a, -1.0, 1.0)
    M = M_min + (a[0] + 1.0) * 0.5 * (M_max - M_min)
    B = B_min + (a[1] + 1.0) * 0.5 * (B_max - B_min)
    K = K_min + (a[2] + 1.0) * 0.5 * (K_max - K_min)
    return float(np.clip(M, M_min, M_max)), \
           float(np.clip(B, B_min, B_max)), \
           float(np.clip(K, K_min, K_max))


def test_trained_model(model_path, xml_path="xml/scene.xml", sim_time=2.0, log_filename="oneleg_eval_log.csv"):
    env = OneLegLandingEnv(xml_path=xml_path, render_mode="human")
    obs, _ = env.reset()
    model = PPO.load(model_path)

    print("=== Start Test ===")

    M_min, M_max = 1.0, 10.0
    B_min, B_max = 0.0, 500.0
    K_min, K_max = 0.0, 2000.0

    max_steps = int(sim_time / (env.dt * env.frame_skip))

    ep_return, ep_len = 0.0, 0
    log_rows = []

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        M, B, K = map_action_to_gains(action, M_min, M_max, B_min, B_max, K_min, K_max)

        q1, q2 = obs[0]*3.14, obs[1]*3.14
        dq1, dq2 = obs[2]*31.4, obs[3]*31.4
        foot_x, foot_z = obs[4]*0.5, obs[5]*0.5
        dfoot_x, dfoot_z = obs[6]*5.0, obs[7]*5.0
        Fz = obs[8]*100.0
        touch = obs[9]
        tau = env.last_tau.copy()

        log_rows.append({
            "t": step * env.dt * env.frame_skip,
            "q1": q1, "q2": q2,
            "dq1": dq1, "dq2": dq2,
            "tau1": tau[0], "tau2": tau[1],
            "foot_x": foot_x, "foot_z": foot_z,
            "dfoot_x": dfoot_x, "dfoot_z": dfoot_z,
            "Fz": Fz, "touch": touch,
            "action0": action[0], "action1": action[1], "action2": action[2],
            "M": M, "B": B, "K": K
        })

        print(f"Step {step}: q=({q1:.3f},{q2:.3f}), dq=({dq1:.3f},{dq2:.3f}), "
              f"tau=({tau[0]:.2f},{tau[1]:.2f}), Fz={Fz:.2f}, touch={touch}")

        time.sleep(env.dt * env.frame_skip)
        ep_return += reward
        ep_len += 1

        if terminated or truncated:
            break

    print(f"Episode finished. return={ep_return:.2f}, length={ep_len}")
    env.close()

    df = pd.DataFrame(log_rows)
    df.to_csv(log_filename, index=False)
    print(f"[Log] Saved to {log_filename}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "proposed_20000000_steps"
    model_path = os.path.join("", "./models", model_name)
    test_trained_model(model_path, sim_time=2.0, log_filename="./data/proposed_0.3m.csv")
