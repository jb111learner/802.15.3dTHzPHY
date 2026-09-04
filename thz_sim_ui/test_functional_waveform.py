import os

import pytest


def test_single_carrier_waveform_only_returns_time_and_spectrum_plots():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_waveform_test

    result = run_waveform_test(spectrum_num_symbols=2048)

    assert result["ok"]
    assert [plot["title"] for plot in result["plots"]] == [
        "单载波时域波形", "功率谱"]
    assert all(plot["png"].startswith(b"\x89PNG") for plot in result["plots"])
    assert not any("PAPR" in plot["title"] for plot in result["plots"])
    assert not any("SC-FDE" in plot["title"] for plot in result["plots"])
    pulse_check = next(
        check for check in result["checks"] if check["name"] == "实部和虚部均包含完整脉冲")
    assert pulse_check["ok"]
    assert "实部 2 个、虚部 2 个" in pulse_check["detail"]


def test_single_carrier_time_and_spectrum_parameters_are_independent():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_waveform_test

    result = run_waveform_test(
        time_filter_length=24,
        time_oversampling=3,
        time_rolloff=0.35,
        time_pulse_count=3,
        spectrum_modulation="16QAM",
        spectrum_filter_length=40,
        spectrum_oversampling=5,
        spectrum_rolloff=0.15,
        spectrum_num_symbols=2048,
    )

    assert result["ok"]
    summary = {item["label"]: item["value"] for item in result["summary"]}
    assert "I/Q 各 3 个脉冲" in summary["时域测试符号"]
    assert "L=24，3×，β=0.35" in summary["时域滤波器"]
    assert summary["功率谱符号"].startswith("16QAM，2048")
    assert "L=40，5×，β=0.15" in summary["功率谱滤波器"]


def test_waveform_ui_defaults_to_single_carrier_typical_parameters():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from thz_sim_ui.pages.functional_test_page import FunctionalTestPage

    app = QApplication.instance() or QApplication([])
    page = FunctionalTestPage()
    try:
        page.test_radios[1].setChecked(True)
        payload = page._build_payload("waveform")
        assert page.wave_mode.currentText() == "单载波"
        assert page.wave_time_symbol.currentText() == "典型复符号"
        assert page.wave_spectrum_modulation.currentText() == "QPSK"
        assert payload["link_mode"] == "sc-fde"
        assert payload["time_pulse_count"] == 2
        assert payload["time_filter_length"] == 32
        assert payload["time_oversampling"] == 4
        assert payload["time_rolloff"] == pytest.approx(0.22)
        assert payload["spectrum_filter_length"] == 32
        assert payload["spectrum_oversampling"] == 4
        assert payload["spectrum_rolloff"] == pytest.approx(0.22)
        assert payload["spectrum_num_symbols"] == 8192
        assert page.wave_ofdm_time_box.isHidden()
        assert page.wave_ofdm_heat_box.isHidden()
        assert page.wave_ofdm_spectrum_box.isHidden()
    finally:
        page.close()
        app.processEvents()


def test_ofdm_single_tone_scan_returns_time_heatmap_and_spectrum_without_papr():
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QImage
    from thz_sim_ui.services.functional_test_service import run_waveform_test

    result = run_waveform_test(link_mode="ofdm")

    assert result["ok"]
    assert [plot["title"] for plot in result["plots"]] == [
        "OFDM 时域波形（含 CP）",
        "OFDM 逐符号频谱热图",
        "OFDM 随机 16QAM 功率谱",
    ]
    assert all(plot["png"].startswith(b"\x89PNG") for plot in result["plots"])
    images = [QImage.fromData(plot["png"], "PNG") for plot in result["plots"]]
    assert all(image.width() >= 2000 and image.height() >= 1200 for image in images)
    assert not any("PAPR" in plot["title"] for plot in result["plots"])
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["OFDM 单音 IFFT 与解析理论波形一致"]["ok"]
    assert checks["OFDM 循环前缀复制正确"]["ok"]
    assert checks["逐符号单音峰值沿指定子载波扫描"]["ok"]
    assert checks["红色带宽边界外开始带外滚降"]["ok"]
    assert checks["随机 16QAM-OFDM 连续时域数据 FFT 有效"]["ok"]
    summary = {item["label"]: item["value"] for item in result["summary"]}
    assert summary["时域扫描子载波"] == "[-6, 5]"
    assert summary["热图扫描范围"] == "k=-24…23，显示 [-28, 27]"
    assert summary["理论占用带宽边界"] == "±15.0 GHz（红色虚线）"
    assert "随机 16QAM，128 个连续" in summary["功率谱数据"]


def test_ofdm_linear_spectrum_and_custom_scan_parameters():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_waveform_test

    result = run_waveform_test(
        link_mode="ofdm",
        ofdm_time_symbol_count=2,
        ofdm_time_start_subcarrier=-12,
        ofdm_time_subcarrier_step=24,
        ofdm_heatmap_symbol_count=16,
        ofdm_heatmap_start_subcarrier=-8,
        ofdm_heatmap_subcarrier_step=1,
        ofdm_heatmap_axis_margin=3,
        ofdm_heatmap_floor_db=-70.0,
        ofdm_spectrum_scale="线性归一化功率",
    )

    assert result["ok"]
    summary = {item["label"]: item["value"] for item in result["summary"]}
    assert summary["时域扫描子载波"] == "[-12, 12]"
    assert summary["热图扫描范围"] == "k=-8…7，显示 [-11, 10]"
    assert summary["功率谱纵坐标"] == "线性归一化功率"


def test_waveform_ui_switches_to_ofdm_parameter_groups_and_payload():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from thz_sim_ui.pages.functional_test_page import FunctionalTestPage

    app = QApplication.instance() or QApplication([])
    page = FunctionalTestPage()
    try:
        page.wave_mode.setCurrentText("OFDM")
        payload = page._build_payload("waveform")
        assert page.wave_sc_time_box.isHidden()
        assert page.wave_sc_spectrum_box.isHidden()
        assert not page.wave_ofdm_time_box.isHidden()
        assert not page.wave_ofdm_heat_box.isHidden()
        assert not page.wave_ofdm_spectrum_box.isHidden()
        assert payload["link_mode"] == "ofdm"
        assert payload["ofdm_time_symbol_count"] == 2
        assert payload["ofdm_time_start_subcarrier"] == -6
        assert payload["ofdm_time_subcarrier_step"] == 11
        assert payload["ofdm_heatmap_symbol_count"] == 48
        assert payload["ofdm_heatmap_floor_db"] == pytest.approx(-45.0)
        assert payload["ofdm_spectrum_scale"] == "对数功率 dB"
        assert payload["ofdm_spectrum_num_symbols"] == 128
        assert payload["ofdm_spectrum_random_seed"] == 2026
        assert not any("papr" in key.lower() for key in payload)
    finally:
        page.close()
        app.processEvents()
