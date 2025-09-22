T = readtable("checkpoint_rewards.csv");
plot(T.steps, T.mean_reward);
hold on;
errorbar(T.steps, T.mean_reward, T.std_reward);
