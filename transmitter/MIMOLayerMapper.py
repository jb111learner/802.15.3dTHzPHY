"""空间层映射：按符号轮转方式把串行符号映射到空间流。"""

from __future__ import annotations

import numpy as np


class MIMOLayerMapper:
    def __init__(self, num_streams):
        self.num_streams = int(num_streams)
        if self.num_streams < 1:
            raise ValueError("num_streams 必须为正整数")

    def map(self, symbols):
        symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
        layers = [symbols[index::self.num_streams].copy() for index in range(self.num_streams)]
        return layers

