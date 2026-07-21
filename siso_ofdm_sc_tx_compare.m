%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OFDM vs SC-FDE 发射波形对比 — 64QAM
%
% 绘制两种波形的时域和频域特性，便于对照 Python 链路系统的发射波形。
% 关键区别：
%   OFDM:  多载波 (IFFT叠加) → 时域类高斯包络 (高PAPR) → 频域矩形谱
%   SC-FDE: 单载波 (RRC脉冲成型) → 时域包络平滑 (低PAPR) → 频域RRC滚降
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;

%% ====================== 公共参数 ======================
MOD_ORDER    = 64;           % 64QAM
SYM_RATE     = 1.0;          % 符号率 (归一化)
SPS          = 4;            % 上采样率 (每符号采样点数)
ROLLOFF      = 0.22;         % RRC 滚降系数
FILTER_LEN   = 32;           % RRC 滤波器半符号长度
N_DATA_SYMS  = 1000;         % 数据符号数 (对比用)

% OFDM 参数
N_SC         = 512;          % 子载波数
CP_LEN       = 32;           % CP 长度
N_OFDM_SYMS  = 20;           % OFDM 符号数
PILOT_IDX    = [1 17 33];    % 块状导频索引 (1-based)

% 绘图参数
FS_COLOR_OFDM = [0.00 0.45 0.74];   % 蓝色
FS_COLOR_SC   = [0.85 0.33 0.10];   % 橙色

%% ====================== 64QAM 星座 ======================
modvec_64qam = (1/sqrt(42)) .* [-7 -5 -1 -3 +7 +5 +1 +3];

% 生成随机 64QAM 符号
rng(42);
tx_data = randi(MOD_ORDER, 1, N_DATA_SYMS) - 1;
mod_fcn_64qam = @(x) complex(modvec_64qam(1+bitshift(x, -3)), ...
                             modvec_64qam(1+mod(x, 8)));
tx_syms = arrayfun(mod_fcn_64qam, tx_data);
tx_syms = tx_syms ./ sqrt(mean(abs(tx_syms).^2));   % 功率归一化

fprintf('64QAM 符号数: %d,  平均功率: %.4f\n', N_DATA_SYMS, mean(abs(tx_syms).^2));

%% ====================== OFDM 发射波形 ======================
fprintf('\n--- 生成 OFDM 波形 ---\n');

% 1. 数据分帧 (每 OFDM 符号的数据子载波数)
n_data_sc = N_SC - length(PILOT_IDX);
n_ofdm_syms_needed = ceil(length(tx_syms) / n_data_sc);
% 补零到整数个 OFDM 符号
pad_len = n_ofdm_syms_needed * n_data_sc - length(tx_syms);
tx_syms_padded = [tx_syms, zeros(1, pad_len)];

% 2. 构造 OFDM 频域网格
data_sc_idx = setdiff(1:N_SC, PILOT_IDX);
tx_syms_mat = reshape(tx_syms_padded, n_data_sc, n_ofdm_syms_needed);

ofdm_freq_grid = zeros(N_SC, n_ofdm_syms_needed);
ofdm_freq_grid(data_sc_idx, :) = tx_syms_mat;
% 导频：BPSK (±1)
pilot_seq = (2*randi(2, N_SC, 1) - 3);
for p = PILOT_IDX
    ofdm_freq_grid(p, :) = pilot_seq(p);
end

% 3. IFFT → 时域
ofdm_time = ifft(ofdm_freq_grid, N_SC, 1);  % (N_SC, n_ofdm_syms)

% 4. 加 CP
ofdm_with_cp = [ofdm_time(end-CP_LEN+1:end, :); ofdm_time];  % (N_SC+CP_LEN, n_ofdm_syms)
ofdm_tx_sym_rate = ofdm_with_cp(:).';  % 1D 符号率信号

% 5. 前导码 (简化：用 Zadoff-Chu 或重复BPSK)
preamble_sym = exp(1i * pi/4 * (0:5119));      % π/2-BPSK 前导码 (行向量)
preamble_sym = preamble_sym ./ sqrt(mean(abs(preamble_sym).^2));
ofdm_full_sym = [preamble_sym, ofdm_tx_sym_rate];

% 6. 上采样 + RRC 成型
ofdm_up = upsample_manual(ofdm_full_sym, SPS);
rrc_filter = rcosdesign(ROLLOFF, FILTER_LEN, SPS, 'sqrt');
ofdm_tx = conv(ofdm_up, rrc_filter, 'same');

fprintf('  OFDM 信号长度 (过采样): %d 样点\n', length(ofdm_tx));
fprintf('  OFDM 每子帧符号数: %d (数据 %d + 导频 %d)\n', ...
    N_SC, n_data_sc, length(PILOT_IDX));

%% ====================== SC-FDE 发射波形 ======================
fprintf('\n--- 生成 SC-FDE 波形 ---\n');

% 1. 分块 (每块 N_SC 个符号)
sc_block_len = N_SC;
n_sc_blocks = ceil(length(tx_syms) / sc_block_len);
pad_len_sc = n_sc_blocks * sc_block_len - length(tx_syms);
tx_syms_sc = [tx_syms, zeros(1, pad_len_sc)];

sc_blocks = reshape(tx_syms_sc, sc_block_len, n_sc_blocks);

% 2. 加 CP (每块 CP_LEN 个符号)
sc_blocks_with_cp = [sc_blocks(end-CP_LEN+1:end, :); sc_blocks];  % (N_SC+CP_LEN, n_blocks)
sc_tx_sym_rate = sc_blocks_with_cp(:).';

% 3. 前导码 + 数据
sc_full_sym = [preamble_sym, sc_tx_sym_rate];

% 4. 上采样 + RRC 成型 (与 OFDM 相同滤波器)
sc_up = upsample_manual(sc_full_sym, SPS);
sc_tx = conv(sc_up, rrc_filter, 'same');

fprintf('  SC-FDE 信号长度 (过采样): %d 样点\n', length(sc_tx));
fprintf('  SC-FDE 每块: %d 符号 + %d CP\n', N_SC, CP_LEN);

%% ====================== 功率归一化 ======================
% 统一归一化到单位平均功率，便于对比
ofdm_tx = ofdm_tx ./ sqrt(mean(abs(ofdm_tx).^2));
sc_tx   = sc_tx   ./ sqrt(mean(abs(sc_tx).^2));

%% ====================== 频域分析 (PSD, periodogram 平均) ======================
nfft_psd = 4096;
n_avg = 8;
win_len = floor(length(ofdm_tx) / n_avg);
P_ofdm = zeros(nfft_psd, 1);
P_sc   = zeros(nfft_psd, 1);
for k = 1:n_avg
    seg_o = ofdm_tx((k-1)*win_len+1 : k*win_len);
    seg_s = sc_tx((k-1)*win_len+1   : k*win_len);
    P_ofdm = P_ofdm + abs(fftshift(fft(seg_o .* hamming(win_len)', nfft_psd))).^2;
    P_sc   = P_sc   + abs(fftshift(fft(seg_s .* hamming(win_len)', nfft_psd))).^2;
end
P_ofdm = P_ofdm / n_avg / win_len;
P_sc   = P_sc   / n_avg / win_len;
f_psd = (-nfft_psd/2 : nfft_psd/2-1)' / nfft_psd * SPS;

%% ====================== PAPR CCDF ======================
% 逐符号计算 PAPR (每个符号 = SPS 个采样点)
ofdm_papr = zeros(1, floor(length(ofdm_tx)/SPS));
sc_papr   = zeros(1, floor(length(sc_tx)/SPS));
for k = 1:length(ofdm_papr)
    blk = ofdm_tx((k-1)*SPS+1 : k*SPS);
    ofdm_papr(k) = max(abs(blk).^2) / mean(abs(blk).^2);
end
for k = 1:length(sc_papr)
    blk = sc_tx((k-1)*SPS+1 : k*SPS);
    sc_papr(k) = max(abs(blk).^2) / mean(abs(blk).^2);
end

ofdm_papr_db = 10*log10(ofdm_papr);
sc_papr_db   = 10*log10(sc_papr);
% Manual CCDF: sort PAPR values, compute complementary CDF
x_ofdm = sort(ofdm_papr_db);   ccdf_ofdm = 1 - (1:length(x_ofdm))'/length(x_ofdm);
x_sc   = sort(sc_papr_db);     ccdf_sc   = 1 - (1:length(x_sc))'  /length(x_sc);

%% ====================== 可视化 ======================
figure('Position', [100 100 1400 900]);

% ---- (1,1): OFDM 时域波形 (包络) ----
subplot(2,3,1);
n_plot = min(3000, length(ofdm_tx));
plot((1:n_plot)/SPS, abs(ofdm_tx(1:n_plot)), 'Color', FS_COLOR_OFDM, 'LineWidth', 0.6);
hold on;
plot((1:n_plot)/SPS, real(ofdm_tx(1:n_plot)), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.3);
yline(mean(abs(ofdm_tx)), 'r--', 'LineWidth', 1);
xlabel('Symbol index'); ylabel('Amplitude');
title(sprintf('OFDM Time-Domain Envelope (PAPR=%.1f dB)', ...
    max(ofdm_papr_db)));
legend('|envelope|', 'Re', 'mean', 'Location', 'best');
grid on; ylim([0 4]);

% ---- (1,2): SC-FDE 时域波形 (包络) ----
subplot(2,3,2);
n_plot_sc = min(3000, length(sc_tx));
plot((1:n_plot_sc)/SPS, abs(sc_tx(1:n_plot_sc)), 'Color', FS_COLOR_SC, 'LineWidth', 0.6);
hold on;
plot((1:n_plot_sc)/SPS, real(sc_tx(1:n_plot_sc)), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.3);
yline(mean(abs(sc_tx)), 'r--', 'LineWidth', 1);
xlabel('Symbol index'); ylabel('Amplitude');
title(sprintf('SC-FDE Time-Domain Envelope (PAPR=%.1f dB)', ...
    max(sc_papr_db)));
legend('|envelope|', 'Re', 'mean', 'Location', 'best');
grid on; ylim([0 4]);

% ---- (1,3): 时域包络叠加对比 ----
subplot(2,3,3);
n_comp = min(2000, min(length(ofdm_tx), length(sc_tx)));
plot((1:n_comp)/SPS, abs(ofdm_tx(1:n_comp)), 'Color', FS_COLOR_OFDM, ...
    'LineWidth', 1.2); hold on;
plot((1:n_comp)/SPS, abs(sc_tx(1:n_comp)), 'Color', FS_COLOR_SC, ...
    'LineWidth', 1.2);
xlabel('Symbol index'); ylabel('|Envelope|');
title('Envelope Comparison (OFDM vs SC-FDE)');
legend('OFDM (Gaussian-like)', 'SC-FDE (constant-ish)', 'Location', 'best');
grid on;

% ---- (2,1): OFDM 频谱 (PSD) ----
subplot(2,3,4);
plot(f_psd, 10*log10(P_ofdm), 'Color', FS_COLOR_OFDM, 'LineWidth', 1.2);
xlabel('Normalized Frequency (× F_s)'); ylabel('PSD (dB)');
title('OFDM Spectrum (Rectangular)');
grid on; xlim([-0.6 0.6]); ylim([-40 10]);

% ---- (2,2): SC-FDE 频谱 (PSD) ----
subplot(2,3,5);
plot(f_psd, 10*log10(P_sc), 'Color', FS_COLOR_SC, 'LineWidth', 1.2);
xlabel('Normalized Frequency (× F_s)'); ylabel('PSD (dB)');
title(sprintf('SC-FDE Spectrum (RRC roll-off=%.2f)', ROLLOFF));
grid on; xlim([-0.6 0.6]); ylim([-40 10]);

% ---- (2,3): PAPR CCDF ----
subplot(2,3,6);
semilogy(x_ofdm, ccdf_ofdm, 'Color', FS_COLOR_OFDM, 'LineWidth', 1.5); hold on;
semilogy(x_sc,   ccdf_sc,   'Color', FS_COLOR_SC,   'LineWidth', 1.5);
xlabel('PAPR (dB)'); ylabel('CCDF');
title('PAPR CCDF');
legend(sprintf('OFDM (%.1f dB @ 1e-3)', ...
    interp1(ccdf_ofdm, x_ofdm, 1e-3, 'linear', max(x_ofdm))), ...
    sprintf('SC-FDE (%.1f dB @ 1e-3)', ...
    interp1(ccdf_sc, x_sc, 1e-3, 'linear', max(x_sc))), ...
    'Location', 'best');
grid on; ylim([1e-4 1]);

sgtitle(sprintf(['64QAM OFDM vs SC-FDE Transmit Waveform Comparison\\n' ...
    'N_{SC}=%d, CP=%d, SPS=%d, RRC rolloff=%.2f'], ...
    N_SC, CP_LEN, SPS, ROLLOFF), 'FontSize', 14, 'FontWeight', 'bold');

%% ====================== 关键指标输出 ======================
fprintf('\n========== 波形对比摘要 ==========\n');
fprintf('指标                      OFDM          SC-FDE\n');
fprintf('--------------------------------------------------\n');
fprintf('PAPR (max)               %6.1f dB      %6.1f dB\n', ...
    max(ofdm_papr_db), max(sc_papr_db));
fprintf('PAPR (99.9th pctile)     %6.1f dB      %6.1f dB\n', ...
    prctile(ofdm_papr_db, 99.9), prctile(sc_papr_db, 99.9));
fprintf('Avg power               %8.4f        %8.4f\n', ...
    mean(abs(ofdm_tx).^2), mean(abs(sc_tx).^2));
fprintf('Peak power              %8.4f        %8.4f\n', ...
    max(abs(ofdm_tx).^2), max(abs(sc_tx).^2));
fprintf('Crest factor            %6.2f         %6.2f\n', ...
    max(abs(ofdm_tx))/sqrt(mean(abs(ofdm_tx).^2)), ...
    max(abs(sc_tx))/sqrt(mean(abs(sc_tx).^2)));
fprintf('带宽 (3dB)              %.2f×F_s      %.2f×F_s\n', ...
    sum(P_ofdm > max(P_ofdm)/2)/length(f_psd)*(f_psd(end)-f_psd(1)), ...
    sum(P_sc   > max(P_sc)  /2)/length(f_psd)*(f_psd(end)-f_psd(1)));
fprintf('==================================\n');

%% ====================== 辅助函数：上采样 (零插入，不依赖工具箱) ======================
function y = upsample_manual(x, L)
    % 在输入序列的每两个样点之间插入 L-1 个零
    N = length(x);
    y = zeros(1, L * N);
    y(1:L:end) = x;
end

