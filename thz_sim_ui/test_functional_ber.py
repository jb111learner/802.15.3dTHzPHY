import math
import os
from pathlib import Path

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


def test_zero_error_requirement_and_upper_bound_match_target():
    from thz_sim_ui.services.ber_test_service import (
        one_sided_ber_upper,
        required_zero_error_bits,
    )

    bits = required_zero_error_bits(1e-6, 0.95)
    assert bits == math.ceil(-math.log(0.05) / 1e-6)
    assert one_sided_ber_upper(0, bits, 0.95) <= 1e-6
    assert one_sided_ber_upper(0, bits - 1, 0.95) > 1e-6
    assert one_sided_ber_upper(10, 1_000_000, 0.95) > 1e-5


def test_six_required_modes_build_expected_ldpc_rs_awgn_params():
    from thz_sim_ui.services.ber_test_service import BER_CONFIGS, build_ber_params

    assert [item["key"] for item in BER_CONFIGS] == [
        "64qam_ofdm", "16qam_ofdm", "64qam_sc", "16qam_sc",
        "64qam_ofdm_rs", "64qam_sc_rs",
    ]
    for config in BER_CONFIGS:
        params = build_ber_params(config, config["snr_values"][0], 2026)
        if config["code_type"] == "LDPC":
            assert params.get("ldpc_matrix_type") == "ieee802153d_1440"
            assert params.get("ldpc_standard_rate") == "14/15"
            assert params.get("ldpc_n") == 1440
            assert params.get("ldpc_k") == 1344
        else:
            assert params.get("rs_nsym") == config["rs_nsym"]
            assert params.get("rs_c_exp") == config["rs_c_exp"]
            assert params.get("rs_packet_size") == config["rs_packet_size"]
            assert params.get("decode_mode") == "hard"
        assert params.get("code_type") == config["code_type"]
        assert params.get("enable_awgn") is True
        assert params.get("enable_multipath") is False
        assert params.get("enable_cfo") is False
        assert params.get("enable_iq_imbalance") is False


def test_ber_runner_scans_verifies_and_returns_plot_without_phy_runtime(tmp_path):
    from thz_sim_ui.services.ber_test_service import run_ber_test

    configs = ({
        "key": "fake", "label": "测试模式", "mode": "ofdm", "ncbps": 4,
        "snr_values": (0, 1, 2),
    },)

    def fake_frame(config, snr_db, seed):
        # 每帧 100 bit：0 dB 预扫描很快因误码停止；1 dB 零误码并完成验证。
        return {
            "bits": 100, "errors": 10 if snr_db == 0 else 0,
            "elapsed_seconds": 0.001, "EbN0_dB": snr_db - 1.0,
        }

    result = run_ber_test(
        target_ber=0.02, confidence=0.95, quick_max_bits=100,
        quick_min_errors=5, save_artifacts=False, output_root=str(tmp_path),
        configs=configs, frame_runner=fake_frame,
    )
    assert result["ok"]
    assert result["plots"][0]["png"].startswith(b"\x89PNG")
    config_result = result["data"]["results"][0]
    assert config_result["threshold_SNRdB"] == 1.0
    assert [point["SNRdB"] for point in config_result["points"]] == [0.0, 1.0]
    verified = next(point for point in config_result["points"] if point["passes"])
    assert verified["total_bits"] >= result["data"]["minimum_verification_bits"]
    assert verified["ber_upper_95"] <= 0.02


def test_ber_modes_cover_batch_and_single_projects_with_snr_defaults():
    from thz_sim_ui.services.ber_test_service import (
        BER_TEST_MODES,
        ber_mode_schemes,
        ber_mode_snr_default,
        build_mode_configs,
    )

    assert [mode["key"] for mode in BER_TEST_MODES] == [
        "modulation_compare", "codec_compare", "waveform_compare", "tbps05",
    ]
    # 前三种方式默认 SNR 范围取自 batch 工程；0.5Tbps 为 35～45 dB、步进 1 dB
    assert ber_mode_snr_default("modulation_compare") == (14.0, 23.0, 1.0)
    assert ber_mode_snr_default("codec_compare") == (18.0, 23.0, 1.0)
    assert ber_mode_snr_default("waveform_compare") == (18.0, 23.0, 1.0)
    assert ber_mode_snr_default("tbps05") == (35.0, 45.0, 1.0)

    # 对比方式各含两条链路；0.5Tbps 为单链路 256QAM MIMO OFDM LDPC
    assert [scheme["project"] for scheme in ber_mode_schemes("modulation_compare")] == [
        "OFDM_64QAM_LDPC(1056,1440)_AWGN", "OFDM_16QAM_LDPC(1056,1440)_AWGN",
    ]
    assert [scheme["project"] for scheme in ber_mode_schemes("tbps05")] == [
        "256QAM_MIMO_OFDM_AWGN_LDPC",
    ]

    configs = build_mode_configs("tbps05", 35.0, 37.0, 1.0)
    assert configs[0]["snr_values"] == (35.0, 36.0, 37.0)
    mapped = configs[0]["mapped"]
    assert mapped["enable_mimo"] is True
    assert mapped["num_spatial_streams"] == 2
    assert mapped["NCBPS"] == 8
    assert mapped["code_type"] == "LDPC"
    assert mapped["ldpc_standard_rate"] == "11/15"
    # 0.5Tbps 保留工程实测信道并固定为确定性回放（实测抽头按 30 GHz 采样率对齐）
    assert configs[0]["keep_channel"] is True
    assert mapped["enable_multipath"] is True
    assert mapped["multipath_source"] == "measured"
    assert mapped["measured_channel_mode"] == "deterministic"
    assert mapped["measured_channel_scenario"] == "50cm"
    assert mapped["sample_rate"] == 30e9
    # 链路配置显示同样按 BER 口径修正为确定性回放
    assert (ber_mode_schemes("tbps05")[0]["config"]["_channel_params"]
            ["measured_channel_mode"] == "deterministic")


def test_build_ber_params_channel_per_mode():
    from thz_sim_ui.services.ber_test_service import build_ber_params, build_mode_configs

    # 对比方式：仅 AWGN 口径（多径关闭、MIMO 恒等信道）
    awgn_cfg = build_mode_configs("modulation_compare", 14.0, 14.0, 1.0)[0]
    params = build_ber_params(awgn_cfg, 14.0, 2026)
    assert params.get("enable_multipath") is False

    # 0.5Tbps：保留实测确定性回放信道，仅叠加 AWGN 与扫描 SNR；数据量按
    # 工程配置时长（256QAM_MIMO_OFDM_AWGN_LDPC 为 0.05 ms → 5e-5 s）
    measured_cfg = build_mode_configs("tbps05", 40.0, 40.0, 1.0)[0]
    params = build_ber_params(measured_cfg, 40.0, 2026)
    assert params.get("enable_multipath") is True
    assert params.get("multipath_source") == "measured"
    assert params.get("measured_channel_mode") == "deterministic"
    assert params.get("enable_awgn") is True
    assert params.get("SNRdB") == 40.0
    assert params.get("duration") == 5e-5


def test_comparison_title_and_saved_curves(tmp_path):
    from thz_sim_ui.services.ber_test_service import (
        comparison_title,
        generate_saved_curves,
    )

    assert comparison_title(["64QAM_LDPC(1056,1440)_OFDM",
                             "64QAM_LDPC(1056,1440)_SC"]) == "64QAM_LDPC_(OFDM vs SC)"
    assert comparison_title(["64QAM_LDPC(1056,1440)_OFDM",
                             "16QAM_LDPC(1056,1440)_OFDM"]) == "(64QAM vs 16QAM)_LDPC_OFDM"
    assert comparison_title(["64QAM_LDPC(1056,1440)_OFDM",
                             "64QAM_RS(11,15)_OFDM"]) == "64QAM_(LDPC vs RS)_OFDM"
    # 0.5Tbps 实际使用实测信道，标题剔除工程名残留的 AWGN 字样
    assert comparison_title(["256QAM_MIMO_OFDM_AWGN_LDPC"]) == "256QAM_MIMO_OFDM_LDPC"
    assert comparison_title([]) == "全链路 BER 曲线"

    document = {
        "target_BER": 1e-6,
        "results": [{
            "label": "64QAM_LDPC(1056,1440)_OFDM",
            "points": [{"SNRdB": 20.0, "EbN0_dB": 14.6, "BER": 1e-4, "total_errors": 10,
                        "total_bits": 100000, "ber_upper_95": 1.6e-4, "passes": False},
                       {"SNRdB": 21.0, "EbN0_dB": 15.6, "BER": 0.0, "total_errors": 0,
                        "total_bits": 100000, "ber_upper_95": 3e-5, "passes": True}],
        }],
    }
    import json
    json_path = tmp_path / "ber_results.json"
    json_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    outputs = generate_saved_curves(str(json_path))
    assert outputs == [str(tmp_path / "ber_curves.png"),
                       str(tmp_path / "ber_curves_ebn0.png")]
    for output in outputs:
        assert Path(output).read_bytes().startswith(b"\x89PNG")


def test_ber_ui_is_last_option_and_builds_payload():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from thz_sim_ui.pages.functional_test_page import FunctionalTestPage

    app = QApplication.instance() or QApplication([])
    page = FunctionalTestPage()
    try:
        page.test_radios[-1].setChecked(True)
        app.processEvents()
        assert page._selected_test() == "ber"
        assert not page.stop_button.isHidden()
        # 默认「对比调制方式」：SNR 范围默认值取自 batch 工程 (14～23 dB, 步进 1)
        payload = page._build_payload("ber")
        assert payload == {
            "mode_key": "modulation_compare",
            "snr_min": 14.0,
            "snr_max": 23.0,
            "snr_step": 1.0,
            "quick_max_bits": 300000,
            "quick_min_errors": 100,
            "random_seed": 2026,
        }
        assert "64QAM_LDPC(1056,1440)_OFDM" in page.ber_link_config.toPlainText()
        # 切换到 0.5Tbps 方式：显示单链路配置且 SNR 范围恢复为 35～45 dB
        index = page.ber_mode_combo.findData("tbps05")
        assert index >= 0
        page.ber_mode_combo.setCurrentIndex(index)
        app.processEvents()
        assert page.ber_snr_min.value() == 35.0
        assert page.ber_snr_max.value() == 45.0
        assert page.ber_snr_step.value() == 1.0
        assert "256QAM_MIMO_OFDM_AWGN_LDPC" in page.ber_link_config.toPlainText()
    finally:
        page.close()
        app.processEvents()
