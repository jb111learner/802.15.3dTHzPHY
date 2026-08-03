def collect_receiver_intermediates(receiver) -> dict:
    """收集前端绘图所需的接收机中间结果。"""
    return {
        "rx_matched": getattr(receiver, "rx_matched", None),
        "rx_downsampled": getattr(receiver, "rx_downsampled", None),
        "rx_equalized": getattr(receiver, "rx_equalized", None),
        "rx_iq_compensated": getattr(receiver, "rx_iq_compensated", None),
        "rx_iq_frontend_compensated": getattr(
            receiver, "rx_iq_frontend_compensated", None
        ),
    }
