import os

import pytest


def test_single_carrier_time_waveform_uses_selected_constellation_symbols():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_waveform_time_test

    # QPSK 星座点 0(-1) 与 3(+1)：两个纯实符号，虚部应无脉冲分量
    result = run_waveform_time_test(
        link_mode="sc-fde",
        sc_modulation="QPSK",
        sc_symbol_indexes=(0, 3),
        sc_pulse_count=2,
    )

    assert result["ok"]
    assert [plot["title"] for plot in result["plots"]] == ["单载波时域波形"]
    assert all(plot["png"].startswith(b"\x89PNG") for plot in result["plots"])
    pulse_check = next(
        check for check in result["checks"]
        if check["name"] == "各脉冲 I/Q 分量符合所选星座点")
    assert pulse_check["ok"]
    assert "实部 2/2 个、虚部 0/0 个" in pulse_check["detail"]
    summary = {item["label"]: item["value"] for item in result["summary"]}
    assert summary["时域测试符号"].startswith("QPSK：")


def test_single_carrier_time_waveform_custom_filter_and_pulse_count():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_waveform_time_test

    result = run_waveform_time_test(
        link_mode="sc-fde",
        sc_modulation="16QAM",
        sc_symbol_indexes=(0, 9, 15),
        sc_pulse_count=3,
        sc_filter_type="rrc",
        sc_filter_length=24,
        sc_oversampling=3,
        sc_rolloff=0.35,
    )

    assert result["ok"]
    summary = {item["label"]: item["value"] for item in result["summary"]}
    assert summary["时域测试符号"].startswith("16QAM：")
    assert "L=24，3×，β=0.35" in summary["时域滤波器"]


def test_single_carrier_spectrum_defaults_and_db_scale_checks_pass():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_waveform_spectrum_test

    result = run_waveform_spectrum_test(sc_num_symbols=4096)

    assert result["ok"]
    assert [plot["title"] for plot in result["plots"]] == ["功率谱"]
    assert result["plots"][0]["png"].startswith(b"\x89PNG")
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["频谱通带平坦"]["ok"]
    assert checks["截止频率位置正确"]["ok"]
    assert checks["带外响应符合理论"]["ok"]
    summary = {item["label"]: item["value"] for item in result["summary"]}
    assert summary["功率谱纵坐标"] == "对数功率 dB"


def test_single_carrier_spectrum_custom_parameters_and_linear_scale():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_waveform_spectrum_test

    result = run_waveform_spectrum_test(
        link_mode="sc-fde",
        sc_modulation="16QAM",
        sc_filter_type="rc",
        sc_filter_length=40,
        sc_oversampling=5,
        sc_rolloff=0.15,
        sc_num_symbols=2048,
        sc_scale="线性归一化功率",
    )

    assert result["ok"]
    summary = {item["label"]: item["value"] for item in result["summary"]}
    assert summary["功率谱符号"].startswith("16QAM，2048")
    assert "L=40，5×，β=0.15" in summary["功率谱滤波器"]
    assert summary["功率谱纵坐标"] == "线性归一化功率"


def test_ofdm_time_single_tone_scan_returns_time_and_heatmap():
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QImage
    from thz_sim_ui.services.functional_test_service import run_waveform_time_test

    result = run_waveform_time_test(link_mode="ofdm")

    assert result["ok"]
    assert [plot["title"] for plot in result["plots"]] == [
        "OFDM 时域波形（含 CP）",
        "OFDM 逐符号频谱热图",
    ]
    images = [QImage.fromData(plot["png"], "PNG") for plot in result["plots"]]
    assert all(image.width() >= 2000 and image.height() >= 1200 for image in images)
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["OFDM 单音 IFFT 与解析理论波形一致"]["ok"]
    assert checks["OFDM 循环前缀复制正确"]["ok"]
    assert checks["逐符号单音峰值沿指定子载波扫描"]["ok"]
    summary = {item["label"]: item["value"] for item in result["summary"]}
    assert summary["时域扫描子载波"] == "[-6, 5]"
    assert summary["热图扫描范围"] == "k=-24…23，显示 [-28, 27]"


def test_ofdm_spectrum_linear_scale():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_waveform_spectrum_test

    result = run_waveform_spectrum_test(
        link_mode="ofdm", ofdm_spectrum_scale="线性归一化功率")

    assert result["ok"]
    assert [plot["title"] for plot in result["plots"]] == ["OFDM 随机 16QAM 功率谱"]
    summary = {item["label"]: item["value"] for item in result["summary"]}
    assert summary["功率谱纵坐标"] == "线性归一化功率"
    assert summary["理论占用带宽边界"] == "±15.0 GHz（红色虚线）"
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["红色带宽边界外开始带外滚降"]["ok"]


def test_time_waveform_ui_defaults_to_single_carrier_with_symbol_selectors():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from thz_sim_ui.pages.functional_test_page import FunctionalTestPage

    app = QApplication.instance() or QApplication([])
    page = FunctionalTestPage()
    try:
        page.test_radios[1].setChecked(True)
        assert page.wave_time_mode.currentText() == "单载波"
        assert page.wave_time_sc_modulation.currentText() == "QPSK"
        assert len(page.wave_time_symbol_selectors) == 2
        payload = page._build_payload("waveform_time")
        assert payload["link_mode"] == "sc-fde"
        assert payload["sc_pulse_count"] == 2
        assert payload["sc_symbol_indexes"] == (0, 1)
        assert payload["sc_filter_length"] == 32
        assert payload["sc_oversampling"] == 4
        assert payload["sc_rolloff"] == pytest.approx(0.22)
        assert page.wave_time_ofdm_box.isHidden()
        assert page.wave_time_ofdm_heat_box.isHidden()
        # 切换调制符号后星座点选择范围与显示同步更新
        page.wave_time_sc_modulation.setCurrentText("16QAM")
        app.processEvents()
        assert len(page.wave_time_symbol_selectors) == 2
        payload = page._build_payload("waveform_time")
        assert payload["sc_modulation"] == "16QAM"
        assert payload["sc_symbol_indexes"] == (0, 1)
    finally:
        page.close()
        app.processEvents()


def test_time_waveform_ui_switches_to_ofdm_groups_and_payload():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from thz_sim_ui.pages.functional_test_page import FunctionalTestPage

    app = QApplication.instance() or QApplication([])
    page = FunctionalTestPage()
    try:
        page.wave_time_mode.setCurrentText("OFDM")
        payload = page._build_payload("waveform_time")
        assert page.wave_time_sc_box.isHidden()
        assert page.wave_time_symbol_box.isHidden()
        assert not page.wave_time_ofdm_box.isHidden()
        assert not page.wave_time_ofdm_heat_box.isHidden()
        assert payload["link_mode"] == "ofdm"
        assert payload["ofdm_time_symbol_count"] == 2
        assert payload["ofdm_time_start_subcarrier"] == -6
        assert payload["ofdm_time_subcarrier_step"] == 11
        assert payload["ofdm_heatmap_symbol_count"] == 48
        assert payload["ofdm_heatmap_floor_db"] == pytest.approx(-45.0)
    finally:
        page.close()
        app.processEvents()


def test_spectrum_ui_defaults_and_ofdm_switch():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from thz_sim_ui.pages.functional_test_page import FunctionalTestPage

    app = QApplication.instance() or QApplication([])
    page = FunctionalTestPage()
    try:
        page.test_radios[2].setChecked(True)
        assert page.wave_spectrum_mode.currentText() == "单载波"
        assert page.wave_spectrum_ofdm_box.isHidden()
        payload = page._build_payload("waveform_spectrum")
        assert payload["link_mode"] == "sc-fde"
        assert payload["sc_modulation"] == "QPSK"
        assert payload["sc_num_symbols"] == 8192
        assert payload["sc_scale"] == "对数功率 dB"
        assert payload["sc_filter_length"] == 32

        page.wave_spectrum_mode.setCurrentText("OFDM")
        app.processEvents()
        assert page.wave_spectrum_sc_box.isHidden()
        assert not page.wave_spectrum_ofdm_box.isHidden()
        payload = page._build_payload("waveform_spectrum")
        assert payload["link_mode"] == "ofdm"
        assert payload["ofdm_spectrum_scale"] == "对数功率 dB"
        assert payload["ofdm_spectrum_num_symbols"] == 128
        assert payload["ofdm_spectrum_random_seed"] == 2026
    finally:
        page.close()
        app.processEvents()
