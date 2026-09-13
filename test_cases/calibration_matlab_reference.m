% 校准测试「编码后理论」BER 参考数据生成脚本
%
% 与 Python 功能测试校准（run_function_calibration_test）使用相同仿真配置：
%   - 调制：QPSK（Gray 映射，单位平均功率），信道：AWGN
%   - Eb/N0 定义同 Python：Es/N0 = 2 * rate * Eb/N0（QPSK 每符号 2 编码比特）
%   - LDPC(1440,1056) 码率 11/15：IEEE 802.15.3d 标准 H 矩阵（由 Python
%     utils/LDPCMatrix 导出），ldpcDecode 最小和算法、30 次迭代
%   - RS(255,192)：GF(2^8) 硬判决（官方默认本原多项式 285），译码失败时
%     输出接收到的消息符号（与 Python galois 行为一致）
%
% 全部使用官方 Communications Toolbox 函数：
%   qammod / qamdemod / awgn / ldpcEncode / ldpcDecode /
%   comm.RSEncoder / comm.RSDecoder / biterr
%
% 运行方式：matlab -batch "cd('test_cases'); calibration_matlab_reference"
% 输出：calibration_matlab_reference.json（Python 校准测试自动读取并绘图）

function calibration_matlab_reference()
    repo = fileparts(mfilename('fullpath'));   % test_cases 目录
    S = load(fullfile(repo, 'ieee802153d_1440_H_11_15.mat'), 'H');
    H = sparse(logical(S.H));
    enc_cfg = ldpcEncoderConfig(H);
    % norm-min-sum + 缩放因子 1.0 = 纯最小和，
    % 与 Python LDPCCoder（min_sum，30 次迭代）保持一致
    dec_cfg = ldpcDecoderConfig(H, 'norm-min-sum');
    rng(2026, 'twister');

    ref = struct();
    ref.description = ['编码后理论 BER 参考曲线：与 Python 功能测试校准相同配置 ' ...
        '(QPSK Gray + AWGN, Es/N0 = 2*rate*Eb/N0)'];
    ref.generated_by = 'MATLAB R2024b Communications Toolbox (official functions)';

    % ---------- LDPC(1440,1056) 11/15：Eb/N0 = 1.5..3.0 dB（4 点） ----------
    ldp = struct('name', 'LDPC(1440,1056)', 'code_rate', '11/15', ...
        'decode_algorithm', 'min-sum (ldpcDecode norm-min-sum, scaling=1.0, 30 iterations)');
    ebn0 = [1.5 2 2.5 3];
    ber = zeros(size(ebn0)); nbits = zeros(size(ebn0)); nerr = zeros(size(ebn0));
    for i = 1:numel(ebn0)
        [ber(i), nbits(i), nerr(i)] = simulate_ldpc(enc_cfg, dec_cfg, ebn0(i), 4e6, 500);
        fprintf('LDPC Eb/N0 = %.1f dB : BER = %.3e (%d bits, %d errors)\n', ...
            ebn0(i), ber(i), nbits(i), nerr(i));
    end
    ldp.ebn0_db = ebn0; ldp.ber = ber; ldp.bits = nbits; ldp.errors = nerr;
    ref.ldpc = ldp;

    % ---------- RS(255,192) GF(2^8) 硬判决：Eb/N0 = 4.0..5.5 dB（4 点） ----------
    rs = struct('name', 'RS(255,192)', 'code_rate', '192/255', ...
        'decoding', 'hard-decision (comm.RSDecoder, BitInput)');
    ebn0 = 4:0.5:5.5;
    ber = zeros(size(ebn0)); nbits = zeros(size(ebn0)); nerr = zeros(size(ebn0));
    for i = 1:numel(ebn0)
        [ber(i), nbits(i), nerr(i)] = simulate_rs(ebn0(i), 4e6, 1000);
        fprintf('RS Eb/N0 = %.1f dB : BER = %.3e (%d bits, %d errors)\n', ...
            ebn0(i), ber(i), nbits(i), nerr(i));
    end
    rs.ebn0_db = ebn0; rs.ber = ber; rs.bits = nbits; rs.errors = nerr;
    ref.rs = rs;

    out_file = fullfile(repo, 'calibration_matlab_reference.json');
    text = jsonencode(ref, 'PrettyPrint', true);
    fid = fopen(out_file, 'w');
    fprintf(fid, '%s', text);
    fclose(fid);
    fprintf('saved: %s\n', out_file);
end

% ---------- LDPC 仿真：官方 ldpcEncode / ldpcDecode ----------
function [ber, nbits, nerr] = simulate_ldpc(enc_cfg, dec_cfg, ebn0_db, max_bits, min_errors)
    k = enc_cfg.NumInformationBits;
    rate = enc_cfg.CodeRate;
    esn0_db = 10 * log10(2 * rate) + ebn0_db;

    % LLR 符号约定自检：无噪声下应零误码，否则整体取反
    test_bits = randi([0 1], k, 1);
    test_cw = ldpcEncode(test_bits, enc_cfg);
    test_s = qammod(test_cw, 4, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);
    test_llr = qamdemod(test_s, 4, 'gray', 'OutputType', 'llr', ...
        'UnitAveragePower', true, 'NoiseVariance', 1e-9);
    flip_llr = biterr(test_bits, ...
        ldpcDecode(test_llr, dec_cfg, 30, MinSumScalingFactor=1.0)) > k / 2;

    nerr = 0; nbits = 0;
    while nbits < max_bits && nerr < min_errors
        blk = randi([0 1], k, 1);
        cw = ldpcEncode(blk, enc_cfg);
        s = qammod(cw, 4, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);
        y = awgn(s, esn0_db, 'measured');
        llr = qamdemod(y, 4, 'gray', 'OutputType', 'llr', ...
            'UnitAveragePower', true, 'NoiseVariance', 10^(-esn0_db / 10));
        if flip_llr
            llr = -llr;
        end
        decoded = ldpcDecode(llr, dec_cfg, 30, MinSumScalingFactor=1.0);
        nerr = nerr + biterr(blk, decoded);
        nbits = nbits + k;
    end
    ber = nerr / nbits;
end

% ---------- RS 仿真：官方 comm.RSEncoder / comm.RSDecoder ----------
function [ber, nbits, nerr] = simulate_rs(ebn0_db, max_bits, min_errors)
    n = 255; k = 192;
    rate = k / n;
    esn0_db = 10 * log10(2 * rate) + ebn0_db;
    enc = comm.RSEncoder(n, k, 'BitInput', true);
    dec = comm.RSDecoder(n, k, 'BitInput', true);

    nerr = 0; nbits = 0;
    while nbits < max_bits && nerr < min_errors
        msg = randi([0 1], k * 8, 1);
        cw = enc(msg);
        s = qammod(cw, 4, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);
        y = awgn(s, esn0_db, 'measured');
        rx = qamdemod(y, 4, 'gray', 'OutputType', 'bit', 'UnitAveragePower', true);
        decoded = dec(rx);
        nerr = nerr + biterr(msg, decoded);
        nbits = nbits + k * 8;
    end
    ber = nerr / nbits;
end
