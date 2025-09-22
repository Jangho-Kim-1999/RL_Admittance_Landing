% CSV 읽기
T = readtable("adm_logging_0.3.csv");

figure;

% 1. 각도
subplot(3,3,1);
plot(T.t, T.q1, 'r', 'LineWidth',1.5); hold on;
plot(T.t, T.q2, 'b', 'LineWidth',1.5);
ylabel("Angle [rad]"); legend("q1","q2"); grid on;

% 2. 각속도
subplot(3,3,2);
plot(T.t, T.dq1, 'r', 'LineWidth',1.5); hold on;
plot(T.t, T.dq2, 'b', 'LineWidth',1.5);
ylabel("Ang vel [rad/s]"); legend("dq1","dq2"); grid on;

% 3. 토크
subplot(3,3,3);
plot(T.t, T.tau1, 'r', 'LineWidth',1.5); hold on;
plot(T.t, T.tau2, 'b', 'LineWidth',1.5);
ylabel("Torque [Nm]"); legend("tau1","tau2"); grid on;

% 4. Foot 위치
subplot(3,3,4);
plot(T.t, T.foot_x, 'r', 'LineWidth',1.5); hold on;
plot(T.t, T.foot_z, 'b', 'LineWidth',1.5);
ylabel("Foot pos [m]"); legend("x","z"); grid on;

% 5. Foot 속도
subplot(3,3,5);
plot(T.t, T.dfoot_x, 'r', 'LineWidth',1.5); hold on;
plot(T.t, T.dfoot_z, 'b', 'LineWidth',1.5);
ylabel("Foot vel [m/s]"); legend("dx","dz"); grid on;

% 6. GRF + Touch
subplot(3,3,6);
yyaxis left;
plot(T.t, T.Fz, 'k', 'LineWidth',1.5);
ylabel("GRF [N]");
yyaxis right;
stairs(T.t, T.touch, 'g', 'LineWidth',1.5);
ylabel("Touch flag"); ylim([-0.2 1.2]);
grid on; legend("Fz","Touch");
xlabel("Time [s]");

% 7. M
subplot(3,3,7);
plot(T.t, T.M, 'm', 'LineWidth',1.5);
ylabel("M [kg]"); grid on;

% 8. B
subplot(3,3,8);
plot(T.t, T.B, 'c', 'LineWidth',1.5);
ylabel("B [Ns/m]"); grid on;

% 9. K
subplot(3,3,9);
plot(T.t, T.K, 'Color',[0.5 0.2 0.8], 'LineWidth',1.5);
ylabel("K [N/m]"); grid on;

sgtitle("One-Leg Stand Logging Results (q, dq, tau, foot, Fz, gains M/B/K)");
