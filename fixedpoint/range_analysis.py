"""可合并范围统计，确定性优先级采样不消耗链路随机数。"""
import numpy as np


class RangeStatistics:
    def __init__(self, reservoir_size=4096):
        if int(reservoir_size) < 1:
            raise ValueError("reservoir_size must be positive")
        self.reservoir_size = int(reservoir_size)
        self.count = 0
        self.nonfinite_count = 0
        self.minimum = None
        self.maximum = None
        self.mean = 0.0
        self.m2 = 0.0
        self.samples = np.empty(0)
        self.priorities = np.empty(0, dtype=np.uint64)

    @staticmethod
    def _priority(indices):
        # SplitMix64；这里的 uint64 模运算仅用于采样哈希。
        values = indices + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))

    def observe(self, value):
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        finite = np.isfinite(values)
        self.nonfinite_count += int(np.count_nonzero(~finite))
        values = values[finite]
        if not values.size:
            return
        size = values.size
        old_count = self.count
        mean = float(values.mean())
        centered = values - mean
        m2 = float(np.dot(centered, centered))
        delta = mean - self.mean
        total = old_count + size
        self.m2 += m2 + delta * delta * old_count * size / total
        self.mean += delta * size / total
        self.count = total
        minimum, maximum = float(values.min()), float(values.max())
        self.minimum = minimum if self.minimum is None else min(self.minimum, minimum)
        self.maximum = maximum if self.maximum is None else max(self.maximum, maximum)
        priorities = self._priority(np.arange(old_count, total, dtype=np.uint64))
        if size > self.reservoir_size:
            indices = np.argpartition(priorities, self.reservoir_size - 1)[:self.reservoir_size]
            values, priorities = values[indices], priorities[indices]
        self.samples = np.concatenate((self.samples, values))
        self.priorities = np.concatenate((self.priorities, priorities))
        if self.samples.size > self.reservoir_size:
            indices = np.argpartition(self.priorities, self.reservoir_size - 1)[:self.reservoir_size]
            self.samples, self.priorities = self.samples[indices], self.priorities[indices]

    def merge(self, other):
        if other.count:
            total = self.count + other.count
            delta = other.mean - self.mean
            self.m2 += other.m2 + delta * delta * self.count * other.count / total
            self.mean += delta * other.count / total
            self.count = total
            self.minimum = other.minimum if self.minimum is None else min(self.minimum, other.minimum)
            self.maximum = other.maximum if self.maximum is None else max(self.maximum, other.maximum)
            # 保留原优先级。重新为两个等长reservoir赋等权hash会忽略源count，
            # 将百万样本与万样本错误地按1:1混合。原优先级的截断阈值
            # 随源count缩小，因此保留键后继续选全局最小K项。
            samples = np.concatenate((self.samples, other.samples))
            priorities = np.concatenate((self.priorities, other.priorities))
            indices = np.argsort(priorities)[:self.reservoir_size]
            self.samples, self.priorities = samples[indices], priorities[indices]
        self.nonfinite_count += other.nonfinite_count

    def report(self):
        percentiles = (0.1, 1, 50, 99, 99.9, 99.99)
        return {"count": self.count, "nonfinite_count": self.nonfinite_count,
                "min": self.minimum, "max": self.maximum, "mean": self.mean if self.count else None,
                "std": float(np.sqrt(self.m2 / self.count)) if self.count else None,
                "percentile": {str(p): float(np.percentile(self.samples, p))
                               for p in percentiles} if self.samples.size else {},
                "percentile_method": "deterministic_priority_reservoir",
                "percentile_approximate": self.count > self.samples.size,
                "percentile_sample_count": int(self.samples.size),
                "percentile_tail_resolution": 1.0 / self.samples.size if self.samples.size else None}


class RangeAnalyzer:
    def __init__(self, enabled=False, reservoir_size=4096):
        self.enabled = bool(enabled)
        self.reservoir_size = int(reservoir_size)
        self.nodes = {}

    def observe(self, node, value):
        if not self.enabled:
            return
        value = np.asarray(value)
        components = {"real": value.real}
        if np.iscomplexobj(value):
            components.update({"imag": value.imag, "magnitude": np.abs(value), "power": np.abs(value) ** 2})
        node_data = self.nodes.setdefault(str(node), {})
        for component, data in components.items():
            node_data.setdefault(component, RangeStatistics(self.reservoir_size)).observe(data)

    def report(self):
        return {node: {component: stats.report() for component, stats in components.items()}
                for node, components in self.nodes.items()}
