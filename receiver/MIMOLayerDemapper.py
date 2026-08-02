"""空间层反映射。"""

from __future__ import annotations

import numpy as np


class MIMOLayerDemapper:
    @staticmethod
    def demap(layers, original_symbol_count):
        arrays = [np.asarray(layer).reshape(-1) for layer in layers]
        if not arrays:
            return np.array([], dtype=np.complex128)
        count = int(original_symbol_count)
        output = np.empty(count, dtype=np.result_type(*arrays))
        num_streams = len(arrays)
        for index in range(count):
            stream_index = index % num_streams
            offset = index // num_streams
            if offset >= len(arrays[stream_index]):
                raise ValueError("空间流长度不足，无法恢复原始符号顺序")
            output[index] = arrays[stream_index][offset]
        return output

