%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 纯软件仿真 OFDM 收发机 (SISO)
% 运行方式：直接执行
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;

%% ====================== 参数配置 ======================
% 波形参数
N_OFDM_SYMS             = 500;          % OFDM 符号数
MOD_ORDER               = 16;           % 调制阶数 {2,4,16,64}
TX_SCALE                = 1.0;          % 发送幅度归一化因子

% OFDM 参数
SC_IND_PILOTS           = [8 22 44 58];                           % 导频子载波
SC_IND_DATA             = [2:7 9:21 23:27 39:43 45:57 59:64];     % 数据子载波
N_SC                    = 64;                                     % 子载波总数
CP_LEN                  = 16;                                     % 循环前缀长度
N_DATA_SYMS             = N_OFDM_SYMS * length(SC_IND_DATA);      % 总数据符号数

% 插值参数 (必须为2)
INTERP_RATE             = 2;

% 接收处理参数
FFT_OFFSET              = 4;           % FFT 窗口偏移 (从 CP 中取多少样点)
LTS_CORR_THRESH         = 0.8;         % LTS 相关检测门限
DO_APPLY_CFO_CORRECTION = 1;           % 是否估计并校正 CFO
DO_APPLY_PHASE_ERR_CORRECTION = 1;     % 是否校正残余相位误差
DO_APPLY_SFO_CORRECTION = 1;           % 是否校正采样频偏 (SFO)

% 仿真信道参数 (可自行调整)
SNR_dB                  = inf;         % 信噪比 (dB)，inf 表示无噪声
CFO_Hz                  = 0;           % 人为加入的载波频偏 (Hz)
SFO_ppm                 = 0;           % 人为加入的采样频偏 (ppm)

% 系统参数
SAMP_FREQ               = 40e6;        % 基带采样率 (Hz)
Ts                      = 1/SAMP_FREQ;

%% ====================== 半带插值滤波器 (2x) ======================
interp_filt2 = zeros(1,43);
interp_filt2([1 3 5 7 9 11 13 15 17 19 21]) = ...
    [12 -32 72 -140 252 -422 682 -1086 1778 -3284 10364];
interp_filt2([23 25 27 29 31 33 35 37 39 41 43]) = ...
    interp_filt2(fliplr([1 3 5 7 9 11 13 15 17 19 21]));
interp_filt2(22) = 16384;
interp_filt2 = interp_filt2 ./ max(abs(interp_filt2));

%% ====================== 生成前导码 (STS + LTS) ======================
% STS (短训练序列) —— 用于 AGC 和粗同步
sts_f = zeros(1,64);
sts_f(1:27) = [0 0 0 0 -1-1i 0 0 0 -1-1i 0 0 0 1+1i 0 0 0 1+1i 0 0 0 1+1i 0 0 0 1+1i 0 0];
sts_f(39:64) = [0 0 1+1i 0 0 0 -1-1i 0 0 0 1+1i 0 0 0 -1-1i 0 0 0 -1-1i 0 0 0 1+1i 0 0 0];
sts_t = ifft(sqrt(13/6) .* sts_f, 64);
sts_t = sts_t(1:16);                    % 取前16个样点作为周期

% LTS (长训练序列) —— 用于 CFO 和信道估计
lts_f = [0 1 -1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 1 1 -1 -1 1 -1 1 -1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1];
lts_t = ifft(lts_f, 64);

% 前导码 = 30个STS周期 + LTS的CP(32样点) + 2个完整的LTS
preamble = [repmat(sts_t, 1, 30)  lts_t(33:64) lts_t lts_t];

%% ====================== 生成数据载荷 ======================
% 随机整数 (0 ~ MOD_ORDER-1)
tx_data = randi(MOD_ORDER, 1, N_DATA_SYMS) - 1;

% 调制映射 (避免依赖通信工具箱)
modvec_bpsk   =  (1/sqrt(2))  .* [-1 1];
modvec_16qam  =  (1/sqrt(10)) .* [-3 -1 +3 +1];
modvec_64qam  =  (1/sqrt(43)) .* [-7 -5 -1 -3 +7 +5 +1 +3];

mod_fcn_bpsk  = @(x) complex(modvec_bpsk(1+x),0);
mod_fcn_qpsk  = @(x) complex(modvec_bpsk(1+bitshift(x, -1)), modvec_bpsk(1+mod(x, 2)));
mod_fcn_16qam = @(x) complex(modvec_16qam(1+bitshift(x, -2)), modvec_16qam(1+mod(x,4)));
mod_fcn_64qam = @(x) complex(modvec_64qam(1+bitshift(x, -3)), modvec_64qam(1+mod(x,8)));

switch MOD_ORDER
    case 2
        tx_syms = arrayfun(mod_fcn_bpsk, tx_data);
    case 4
        tx_syms = arrayfun(mod_fcn_qpsk, tx_data);
    case 16
        tx_syms = arrayfun(mod_fcn_16qam, tx_data);
    case 64
        tx_syms = arrayfun(mod_fcn_64qam, tx_data);
    otherwise
        error('MOD_ORDER 必须是 2, 4, 16 或 64');
end

% 按 OFDM 符号重排 (每列一个符号)
tx_syms_mat = reshape(tx_syms, length(SC_IND_DATA), N_OFDM_SYMS);

% 导频值 (BPSK)
pilots = [1 1 -1 1].';
pilots_mat = repmat(pilots, 1, N_OFDM_SYMS);

%% ====================== IFFT 和 CP 添加 ======================
ifft_in_mat = zeros(N_SC, N_OFDM_SYMS);
ifft_in_mat(SC_IND_DATA, :)   = tx_syms_mat;
ifft_in_mat(SC_IND_PILOTS, :) = pilots_mat;

tx_payload_mat = ifft(ifft_in_mat, N_SC, 1);

% 添加循环前缀
tx_cp = tx_payload_mat((end-CP_LEN+1 : end), :);
tx_payload_mat = [tx_cp; tx_payload_mat];

% 展平为时域向量
tx_payload_vec = reshape(tx_payload_mat, 1, numel(tx_payload_mat));

% 完整基带信号 (前导码 + 载荷)
tx_vec = [preamble tx_payload_vec];

% 补零以适应插值滤波器延迟
tx_vec_padded = [tx_vec, zeros(1, ceil(length(interp_filt2)/2))];

%% ====================== 2x 插值 (上采样+滤波) ======================
tx_vec_2x = zeros(1, 2*numel(tx_vec_padded));
tx_vec_2x(1:2:end) = tx_vec_padded;
tx_vec_air = filter(interp_filt2, 1, tx_vec_2x);

% 归一化到 +/-1
tx_vec_air = TX_SCALE .* tx_vec_air ./ max(abs(tx_vec_air));

fprintf('基带信号长度 (插值后): %d 样点\n', length(tx_vec_air));

%% ====================== 仿真信道 ======================
% 默认理想信道 (接收信号 = 发送信号)
rx_vec_air = tx_vec_air;

% ----- 加性白高斯噪声 (AWGN) -----
if isfinite(SNR_dB)
    % 计算信号功率 (实部+虚部)
    sig_pow = mean(abs(rx_vec_air).^2);
    noise_pow = sig_pow / (10^(SNR_dB/10));
    noise = sqrt(noise_pow/2) * (randn(1,length(rx_vec_air)) + 1i*randn(1,length(rx_vec_air)));
    rx_vec_air = rx_vec_air + noise;
end

% ----- 载波频偏 (CFO) -----
if CFO_Hz ~= 0
    t = (0:length(rx_vec_air)-1) / SAMP_FREQ;
    rx_vec_air = rx_vec_air .* exp(1i*2*pi*CFO_Hz*t);
end

% ----- 采样频偏 (SFO) 和定时偏移 (可选) -----
% 此处未实现，可通过重采样模拟

% 为了模拟触发偏移，在接收信号末尾补零 (但一般不会影响同步)
% 这里略过

%% ====================== 2x 降采样 (匹配滤波+抽取) ======================
raw_rx_dec = filter(interp_filt2, 1, rx_vec_air);
raw_rx_dec = raw_rx_dec(1:2:end);   % 下采样

%% ====================== 同步：LTS 相关检测 ======================
% 互相关 (使用符号函数提高鲁棒性)
lts_corr = abs(conv(conj(fliplr(lts_t)), sign(raw_rx_dec)));

% 跳过首尾部分 (避免边缘假峰)
lts_corr = lts_corr(32:end-32);

% 寻找超过门限的峰值
lts_peaks = find(lts_corr(1:800) > LTS_CORR_THRESH * max(lts_corr));

% 寻找间隔为 length(lts_t) 的两个连续峰值 (对应两个LTS)
[LTS1, LTS2] = meshgrid(lts_peaks, lts_peaks);
[lts_second_peak_index, ~] = find(LTS2 - LTS1 == length(lts_t));

if isempty(lts_second_peak_index)
    error('未找到有效的 LTS 相关峰，同步失败！');
end

% 确定载荷起始索引 (第二个LTS后 + 32样点CP)
payload_ind = lts_peaks(max(lts_second_peak_index)) + 32;
lts_ind = payload_ind - 160;   % LTS 起始位置 (160 = 2.5*64)

%% ====================== CFO 估计与校正 (基于 LTS) ======================
if DO_APPLY_CFO_CORRECTION
    % 提取两个 LTS (未校正 CFO)
    rx_lts = raw_rx_dec(lts_ind : lts_ind+159);
    rx_lts1 = rx_lts(-64 - FFT_OFFSET + [97:160]);   % 第一个64样点
    rx_lts2 = rx_lts(-FFT_OFFSET + [97:160]);        % 第二个64样点

    % 计算相位差，估计 CFO (归一化频率)
    rx_cfo_est_lts = mean(unwrap(angle(rx_lts2 .* conj(rx_lts1))));
    rx_cfo_est_lts = rx_cfo_est_lts / (2*pi*64);      % 归一化到采样率
else
    rx_cfo_est_lts = 0;
end

% 应用 CFO 校正
cfo_corr_t = exp(-1i*2*pi*rx_cfo_est_lts * (0:length(raw_rx_dec)-1));
rx_dec_cfo_corr = raw_rx_dec .* cfo_corr_t;

%% ====================== 信道估计 (基于 LTS) ======================
% 再次提取校正后的 LTS
rx_lts = rx_dec_cfo_corr(lts_ind : lts_ind+159);
rx_lts1 = rx_lts(-64 - FFT_OFFSET + [97:160]);
rx_lts2 = rx_lts(-FFT_OFFSET + [97:160]);

rx_lts1_f = fft(rx_lts1);
rx_lts2_f = fft(rx_lts2);
rx_H_est = lts_f .* (rx_lts1_f + rx_lts2_f)/2;   % 平均后除以发送序列

%% ====================== 数据解调与均衡 ======================
% 提取载荷样点 (整数个 OFDM 符号)
payload_vec = rx_dec_cfo_corr(payload_ind : payload_ind + N_OFDM_SYMS*(N_SC+CP_LEN) - 1);
payload_mat = reshape(payload_vec, (N_SC+CP_LEN), N_OFDM_SYMS);

% 去除 CP，保留 FFT_OFFSET 个样点用于定时偏移补偿
payload_mat_noCP = payload_mat(CP_LEN - FFT_OFFSET + [1:N_SC], :);

% FFT
syms_f_mat = fft(payload_mat_noCP, N_SC, 1);

% 零迫均衡
syms_eq_mat = syms_f_mat ./ repmat(rx_H_est.', 1, N_OFDM_SYMS);

%% ====================== SFO 校正 (基于导频) ======================
if DO_APPLY_SFO_CORRECTION
    % 提取导频符号
    pilots_f_mat = syms_eq_mat(SC_IND_PILOTS, :);
    pilots_f_mat_comp = pilots_f_mat .* pilots_mat;   % 消除调制

    % 计算相位 (注意展开)
    pilot_phases = unwrap(angle(fftshift(pilots_f_mat_comp,1)), [], 1);

    % 求每个符号内相位随子载波的斜率
    pilot_spacing_mat = repmat(mod(diff(fftshift(SC_IND_PILOTS)),64).', 1, N_OFDM_SYMS);
    pilot_slope_mat = mean(diff(pilot_phases) ./ pilot_spacing_mat);

    % 生成补偿相位 (对每个子载波)
    pilot_phase_sfo_corr = fftshift((-32:31).' * pilot_slope_mat, 1);
    pilot_phase_corr = exp(-1i * pilot_phase_sfo_corr);

    % 应用补偿
    syms_eq_mat = syms_eq_mat .* pilot_phase_corr;
else
    pilot_phase_sfo_corr = zeros(N_SC, N_OFDM_SYMS);
end

%% ====================== 残余相位误差校正 (公共相位) ======================
if DO_APPLY_PHASE_ERR_CORRECTION
    % 提取导频并求平均相位
    pilots_f_mat = syms_eq_mat(SC_IND_PILOTS, :);
    pilots_f_mat_comp = pilots_f_mat .* pilots_mat;
    pilot_phase_err = angle(mean(pilots_f_mat_comp));
else
    pilot_phase_err = zeros(1, N_OFDM_SYMS);
end

% 应用相位校正
pilot_phase_err_corr = repmat(pilot_phase_err, N_SC, 1);
pilot_phase_corr = exp(-1i * pilot_phase_err_corr);
syms_eq_pc_mat = syms_eq_mat .* pilot_phase_corr;

% 提取数据子载波
payload_syms_mat = syms_eq_pc_mat(SC_IND_DATA, :);

%% ====================== 解调 ======================
rx_syms = reshape(payload_syms_mat, 1, N_DATA_SYMS);

% 硬判决解调
demod_fcn_bpsk = @(x) double(real(x)>0);
demod_fcn_qpsk = @(x) double(2*(real(x)>0) + 1*(imag(x)>0));
demod_fcn_16qam = @(x) (8*(real(x)>0)) + (4*(abs(real(x))<0.6325)) + ...
                        (2*(imag(x)>0)) + (1*(abs(imag(x))<0.6325));
demod_fcn_64qam = @(x) (32*(real(x)>0)) + (16*(abs(real(x))<0.6172)) + ...
                        (8*((abs(real(x))<0.9258)&&(abs(real(x))>0.3086))) + ...
                        (4*(imag(x)>0)) + (2*(abs(imag(x))<0.6172)) + ...
                        (1*((abs(imag(x))<0.9258)&&(abs(imag(x))>0.3086)));

switch MOD_ORDER
    case 2
        rx_data = arrayfun(demod_fcn_bpsk, rx_syms);
    case 4
        rx_data = arrayfun(demod_fcn_qpsk, rx_syms);
    case 16
        rx_data = arrayfun(demod_fcn_16qam, rx_syms);
    case 64
        rx_data = arrayfun(demod_fcn_64qam, rx_syms);
end

%% ====================== 性能统计 ======================
sym_errs = sum(tx_data ~= rx_data);
bit_errs = sum(dec2bin(bitxor(tx_data, rx_data), log2(MOD_ORDER)) == '1', 'all');
rx_evm   = sqrt(mean(abs(payload_syms_mat(:) - tx_syms_mat(:)).^2));

fprintf('\n========== 结果 ==========\n');
fprintf('数据符号数:  %d\n', N_DATA_SYMS);
fprintf('符号错误:    %d (%.2f%%)\n', sym_errs, 100*sym_errs/N_DATA_SYMS);
fprintf('比特错误:    %d (%.2f%%)\n', bit_errs, 100*bit_errs/(N_DATA_SYMS*log2(MOD_ORDER)));
fprintf('EVM (RMS):   %.2f %%\n', 100*rx_evm);

if DO_APPLY_CFO_CORRECTION
    cfo_est_hz = rx_cfo_est_lts * SAMP_FREQ;
    fprintf('估计 CFO:    %.2f Hz\n', cfo_est_hz);
end

if DO_APPLY_SFO_CORRECTION
    % 粗略估计 SFO (基于导频斜率的变化率)
    drift_sec = pilot_slope_mat / (2*pi*312500);   % 312500 = 20MHz/64
    sfo_est_ppm = 1e6 * mean(diff(drift_sec) / (4e-6));   % 4us 符号间隔
    fprintf('估计 SFO:    %.2f ppm\n', sfo_est_ppm);
end

%% ====================== 绘图 (可选) ======================
% 这里仅绘制星座图作为示例
figure;
plot(real(payload_syms_mat(:)), imag(payload_syms_mat(:)), 'ro', 'MarkerSize', 2);
hold on;
plot(real(tx_syms_mat(:)), imag(tx_syms_mat(:)), 'b.', 'MarkerSize', 2);
axis equal; axis(1.5*[-1 1 -1 1]);
grid on;
title('星座图 (Rx 红, Tx 蓝)');
legend('接收', '发送');

% 可添加更多绘图 (EVM, 信道估计等)，此处略
% 若需要，可参考原始代码中的绘图部分自行添加