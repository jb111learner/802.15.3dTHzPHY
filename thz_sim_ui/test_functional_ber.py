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
            assert params.get("ldpc_standard_rate") == "11/15"
            assert params.get("ldpc_n") == 1440
            assert params.get("ldpc_k") == 1056
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


def test_ber_runner_scans_all_points_strictly_without_verification(tmp_path):
    from thz_sim_ui.services.ber_test_service import run_ber_test

    config = {
        "key": "fake", "label": "测试配置", "mode": "ofdm", "ncbps": 4,
        "snr_values": (0.0, 1.0, 2.0),
    }

    def fake_frame(config, snr_db, seed):
        # 每帧 100 bit：0 dB 有误码；1/2 dB 零误码（无早停、无置信验证）
        return {
            "bits": 100, "errors": 10 if snr_db == 0 else 0,
            "elapsed_seconds": 0.001, "EbN0_dB": snr_db - 1.0,
        }

    result = run_ber_test(
        target_ber=0.02, quick_max_bits=100,
        quick_min_errors=5, save_artifacts=False, output_root=str(tmp_path),
        config=config, frame_runner=fake_frame,
    )
    assert result["ok"]
    assert result["plots"][0]["png"].startswith(b"\x89PNG")
    config_result = result["data"]["results"][0]
    # 严格扫描全部 3 个点：零误码点不再提前结束扫描，也不做置信验证
    assert [point["SNRdB"] for point in config_result["points"]] == [0.0, 1.0, 2.0]
    assert config_result["points"][0]["BER"] == pytest.approx(0.1)
    assert config_result["points"][1]["total_errors"] == 0
    assert config_result["points"][2]["total_errors"] == 0
    for point in config_result["points"]:
        assert "passes" not in point
        assert "verified" not in point
        assert "ber_upper_95" not in point
    # 表格去掉 模式/95% 上限/判定 三列，零误码点显示“无误码”
    assert result["table"]["columns"] == [
        "SNR/dB", "Eb/N0/dB", "误码数", "比特数", "实测 BER", "运行次数"]
    assert result["table"]["rows"][1][4] == "0（无误码）"


def test_ber_options_cover_six_configs_plus_tbps05():
    from thz_sim_ui.services.ber_test_service import (
        BER_CONFIG_OPTIONS,
        ber_option_link_text,
        ber_option_snr_default,
        build_ber_option_config,
    )

    assert [option["key"] for option in BER_CONFIG_OPTIONS] == [
        "64qam_ofdm", "16qam_ofdm", "64qam_sc", "16qam_sc",
        "64qam_ofdm_rs", "64qam_sc_rs", "tbps05",
    ]
    # 六种配置的默认 SNR 范围与旧固定表一致；0.5Tbps 为 35～45 dB、步进 1 dB
    assert ber_option_snr_default("64qam_ofdm") == (14.0, 20.0, 1.0)
    assert ber_option_snr_default("16qam_ofdm") == (9.0, 14.0, 1.0)
    assert ber_option_snr_default("64qam_sc") == (12.0, 18.0, 1.0)
    assert ber_option_snr_default("16qam_sc") == (8.0, 13.0, 1.0)
    assert ber_option_snr_default("64qam_ofdm_rs") == (19.0, 26.0, 1.0)
    assert ber_option_snr_default("64qam_sc_rs") == (16.0, 22.0, 1.0)
    assert ber_option_snr_default("tbps05") == (30.0, 34.0, 1.0)

    config = build_ber_option_config("tbps05", 35.0, 37.0, 1.0)
    assert config["snr_values"] == (35.0, 36.0, 37.0)
    mapped = config["mapped"]
    assert mapped["enable_mimo"] is True
    assert mapped["num_spatial_streams"] == 2
    assert mapped["NCBPS"] == 8
    assert mapped["code_type"] == "LDPC"
    assert mapped["ldpc_standard_rate"] == "11/15"
    # 0.5Tbps 保留工程实测信道并固定为确定性回放（实测抽头按 30 GHz 采样率对齐）
    assert config["keep_channel"] is True
    assert mapped["enable_multipath"] is True
    assert mapped["multipath_source"] == "measured"
    assert mapped["measured_channel_mode"] == "deterministic"
    assert mapped["measured_channel_scenario"] == "50cm"
    assert mapped["sample_rate"] == 30e9

    # 链路配置说明文本：六配置含仅 AWGN 口径，0.5Tbps 含实测确定性回放
    legacy_text = ber_option_link_text("64qam_ofdm")
    assert "64QAM OFDM LDPC" in legacy_text
    assert "仅 AWGN" in legacy_text
    tbps_text = ber_option_link_text("tbps05")
    assert "256QAM_MIMO_OFDM_AWGN_LDPC" in tbps_text
    assert "deterministic" in tbps_text


def test_build_ber_params_channel_per_option():
    from thz_sim_ui.services.ber_test_service import (
        build_ber_option_config,
        build_ber_params,
    )

    # 六种配置：仅 AWGN 口径（多径关闭），短帧逐次累计
    awgn_cfg = build_ber_option_config("64qam_ofdm", 20.0, 20.0, 1.0)
    params = build_ber_params(awgn_cfg, 20.0, 2026)
    assert params.get("enable_multipath") is False
    assert params.get("duration") == 1e-7

    # 0.5Tbps：保留实测确定性回放信道，仅叠加 AWGN 与扫描 SNR；数据量按
    # 工程配置时长（256QAM_MIMO_OFDM_AWGN_LDPC 为 0.05 ms → 5e-5 s）
    measured_cfg = build_ber_option_config("tbps05", 40.0, 40.0, 1.0)
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
        # 默认选中第一条配置（64QAM OFDM LDPC）：SNR 范围 14～20 dB、步进 1
        assert page.ber_config_combo.currentData() == "64qam_ofdm"
        payload = page._build_payload("ber")
        assert payload == {
            "config_key": "64qam_ofdm",
            "snr_min": 14.0,
            "snr_max": 20.0,
            "snr_step": 1.0,
            "quick_max_bits": 3000000,
            "quick_min_errors": 100,
            "random_seed": 2026,
        }
        assert "64QAM OFDM LDPC" in page.ber_link_config.toPlainText()
        # 切换到 0.5Tbps 配置：显示实测信道说明且 SNR 范围恢复为 30～34 dB
        index = page.ber_config_combo.findData("tbps05")
        assert index >= 0
        page.ber_config_combo.setCurrentIndex(index)
        app.processEvents()
        assert page.ber_snr_min.value() == 30.0
        assert page.ber_snr_max.value() == 34.0
        assert page.ber_snr_step.value() == 1.0
        assert "256QAM_MIMO_OFDM_AWGN_LDPC" in page.ber_link_config.toPlainText()
        assert "deterministic" in page.ber_link_config.toPlainText()
    finally:
        page.close()
        app.processEvents()
