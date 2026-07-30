"""
全链路信号数据类型诊断 — AWGN, SC-FDE + OFDM
每步打印信号的实际 dtype 和 shape
"""
import numpy as np
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.THzReceiver import THzReceiver


def show(label, data):
    if data is None:
        print(f"  {label:28s} None")
        return
    if isinstance(data, dict):
        arr = np.asarray(data.get("signal_stream", data))
    else:
        arr = np.asarray(data)
    print(f"  {label:28s} dtype={str(arr.dtype):10s}  shape={str(arr.shape)}")


def run(mode):
    print(f"\n{'='*60}")
    print(f"  {mode.upper()}")
    print(f"{'='*60}")

    params = PHYParams()
    params.update(
        link_mode=mode, SNRdB=24,
        enable_multipath=False, enable_cfo=False,
        enable_phase_noise=False, enable_iq_imbalance=False,
        enable_channel_equalization=True, enable_cfo_compensation=False,
    )

    # TX
    tx = THzTransmitter(params); tx.run()
    show("1.assemble(bits)",      tx.data_bits_dict)
    if tx.is_scramble:
        show("2.scramble(bits)",   tx.data_scrambled_dict)
    show("3.encode(bits)",         tx.coded_bits_dict)
    show("4.modulate(syms)",       tx.modulated_data_dict)
    if mode == "ofdm":
        show("5.ofdm_proc(syms)",   tx.data_ofdm_dict)
    show("6.insert_gi(syms)",      tx.data_with_gi_dict)
    show("7.preamble(syms)",       tx.data_with_preamble_dict)
    show("8.pulse(samples)",       tx.tx_signal_dict)

    # Channel
    ch = THzChannel(params)
    rx = ch.run(tx.tx_signal_dict)
    show("9.channel(samples)",     rx)

    # RX
    recv = THzReceiver(params, tx)
    recv.run(rx)
    show("10.MF(samples)",         recv.rx_matched)
    show("11.coarse_sync(samples)", recv.rx_coarse_synced)
    show("12.cfo_coarse(samples)", recv.rx_cfo_coarse)
    show("13.fine_sync(samples)",  recv.rx_fine_synced)
    show("14.downsample(syms)",    recv.rx_downsampled)
    show("15.cfo_fine(syms)",      recv.rx_cfo_fine)
    show("16.IQ_comp(syms)",       recv.rx_iq_compensated)
    show("17.equalized(syms)",     recv.rx_equalized)
    show("18.LLR(softbits)",       recv.llr_dict)
    show("19.decoded(bits)",       recv.decoded_bits)
    show("20.descrambled(bits)",   recv.data_bits)


if __name__ == "__main__":
    run("sc-fde")
    run("ofdm")
