import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from proposed_train import OneLegLandingEnv

def evaluate_model(model, xml_path="xml/scene.xml", n_eval_episodes=10):
    env = OneLegLandingEnv(xml_path=xml_path, render_mode=None)
    rewards = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            done = terminated or truncated
        rewards.append(ep_ret)
    env.close()
    return np.mean(rewards), np.std(rewards)

def main():
    models_dir = "./models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".zip")]

    step_pattern = re.compile(r"(\d+)_steps\.zip")

    results = []
    for fname in sorted(model_files):
        match = step_pattern.match(fname)
        if not match:
            continue
        steps = int(match.group(1))

        if steps % 1000000 != 0:
            continue
        if steps > 20000001:
            continue

        path = os.path.join(models_dir, fname)
        print(f"Evaluating {fname} ...")
        model = PPO.load(path)
        mean_rew, std_rew = evaluate_model(model)
        results.append((steps, mean_rew, std_rew))
    results.sort(key=lambda x: x[0])


    if not results:
        print("No checkpoints found.")
        return

    df = pd.DataFrame(results, columns=["steps", "mean_reward", "std_reward"])


    csv_path = os.path.join(models_dir, "reward_curve.csv")


    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")



    # Plot
    steps, means, stds = zip(*results)
    plt.figure(figsize=(8,5))
    plt.plot(steps, means, marker="o", label="Mean reward")
    plt.xlabel("Training steps")
    plt.ylabel("Average reward")
    plt.title("Model performance over checkpoints (adm_5th)")
    plt.legend()
    plt.grid(True)
    plt.savefig("checkpoint_rewards.png")
    plt.show()

    # Best model
    best_idx = np.argmax(means)
    print(f"Best model: step={steps[best_idx]}, reward={means[best_idx]:.2f} ± {stds[best_idx]:.2f}")

if __name__ == "__main__":
    main()
