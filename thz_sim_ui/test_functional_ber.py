import math
import os

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
        payload = page._build_payload("ber")
        assert payload == {
            "random_seed": 2026,
            "quick_max_bits": 300000,
            "quick_min_errors": 100,
        }
    finally:
        page.close()
        app.processEvents()
