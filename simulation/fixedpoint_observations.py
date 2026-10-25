"""按原帧元数据分区观察，不改变、重排或重新对齐链路输入。"""
import numpy as np


def observe_transmit_regions(context, transmitter, signal_dict):
    if not context.range_enabled:
        return
    signal = np.asarray(signal_dict["signal_stream"])
    params = transmitter.params
    if params.get("enable_mimo", False):
        metadata = signal_dict["mimo_frame"]
        frame_num = int(metadata["frame_num"])
        guard = int(metadata["guard_length"])
        block_length = int(metadata["block_length"])
        sync_count, training_count = int(metadata["num_sync_symbols"]), int(metadata["num_training_symbols"])
        data_count = int(metadata["num_data_symbols"])
        frame_length = guard + (sync_count + training_count + data_count) * block_length
        if signal.shape[-1] != frame_num * frame_length + guard:
            context.record_coverage("analysis.tx_regions", "metadata_mismatch")
            return
        frames = signal[..., :frame_num * frame_length].reshape(signal.shape[0], frame_num, frame_length)
        context.observe("tx.regions.guard", frames[..., :guard])
        context.observe("tx.regions.final_guard", signal[..., frame_num * frame_length:])
        blocks = frames[..., guard:].reshape(signal.shape[0], frame_num, sync_count + training_count + data_count, block_length)
        cp = int(metadata["cp_length"])
        context.observe("tx.regions.cp", blocks[..., :cp])
        context.observe("tx.regions.sync", blocks[..., :sync_count, cp:])
        context.observe("tx.regions.training", blocks[..., sync_count:sync_count + training_count, cp:])
        context.observe("tx.regions.data", blocks[..., sync_count + training_count:, cp:])
        return
    frame_num = int(transmitter.data_bits_dict["frame_num"])
    if frame_num < 1 or signal.size % frame_num:
        context.record_coverage("analysis.tx_regions", "metadata_mismatch")
        return
    frames = signal.reshape(frame_num, -1)
    oversampling = int(params.get("oversampling"))
    preamble = len(transmitter.preamble) * oversampling
    context.observe("tx.regions.preamble", frames[:, :preamble])
    payload = frames[:, preamble:]
    if params.get("link_mode") == "ofdm":
        length = (int(params.get("subwave_num")) + int(params.get("gi_length"))) * oversampling
        count = int(params.get("subframe_ofdm_num"))
        if payload.shape[-1] != count * length:
            context.record_coverage("analysis.tx_regions", "metadata_mismatch")
            return
        blocks = payload.reshape(frame_num, count, length)
        cp = int(params.get("gi_length")) * oversampling
        pilots = list(params.get("pilot_block_indexes"))
        data = [i for i in range(count) if i not in pilots]
        context.observe("tx.regions.cp", blocks[..., :cp])
        context.observe("tx.regions.pilot", blocks[:, pilots, cp:])
        context.observe("tx.regions.data", blocks[:, data, cp:])
    elif params.get("gi_type") == "cp":
        length = (int(params.get("subframe_length")) + int(params.get("gi_length"))) * oversampling
        if payload.shape[-1] % length:
            context.record_coverage("analysis.tx_regions", "metadata_mismatch")
            return
        blocks = payload.reshape(frame_num, -1, length)
        cp = int(params.get("gi_length")) * oversampling
        context.observe("tx.regions.cp", blocks[..., :cp])
        context.observe("tx.regions.data", blocks[..., cp:])
    else:
        context.observe("tx.regions.payload_with_golay", payload)
