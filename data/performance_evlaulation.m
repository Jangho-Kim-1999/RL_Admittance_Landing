% clc;
% clear;
% close all;


% 로그 불러오기
T = readtable("adm_8th_1100000_steps_1.0.csv");

% 샘플링 시간
dt = mean(diff(T.t));

% -------------------------
% 1. 발 위치 에러 (RMSE)
x_ref = 0.0; z_ref = 0.25;
err_x = T.foot_x - x_ref;
err_z = T.foot_z - z_ref;
err_pos = sqrt(err_x.^2 + err_z.^2);
rmse_foot = sqrt(mean(err_pos.^2));

% -------------------------
% 2. RMS Torque
rms_tau = sqrt(mean(T.tau1.^2 + T.tau2.^2));

% -------------------------
% 3. 총 시간 에너지 소비량
% power = torque * angular velocity
power = abs(T.tau1 .* T.dq1) + abs(T.tau2 .* T.dq2);
energy = sum(power) * dt;   % J (Joule)

% -------------------------
% 4. 피크 GRF
peak_Fz = min(T.Fz);

% -------------------------
% 5. 임펄스
impulse = sum(T.Fz) * dt;   % N·s

% -------------------------
% 결과 출력
fprintf("=== 성능 지표 ===\n");
fprintf("Foot pos RMSE      : %.4f [m]\n", rmse_foot);
fprintf("RMS Torque         : %.4f [Nm]\n", rms_tau);
fprintf("Total Energy usage : %.4f [J]\n", energy);
fprintf("Peak GRF           : %.2f [N]\n", peak_Fz);
fprintf("Impulse            : %.4f [N·s]\n", impulse);
