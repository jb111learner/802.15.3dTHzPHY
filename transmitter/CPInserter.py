import numpy as np

class CPInserter:
    """
    循环前缀（CP）插入器：为每个数据块添加CP，消除多径ISI
    - 支持通过params直接指定数据块长度（N）和CP长度（cp_length）
    - 未指定时自动计算（CP长度为数据块长度的比例或信道最大时延）
    """
    def __init__(self, params):
        self.params = params
        self.cp_ratio = 0.25  # 默认CP长度占数据块长度的比例（未指定cp_length时使用）
        self._init_params()  # 初始化数据块长度和CP长度

    def _init_params(self):
        """初始化数据块长度（N）和CP长度（cp_length）"""
        # 1. 从params获取数据块长度N（优先用户指定，默认64）
        self.N = self.params.get("subframe_length")
        # 2. 从params获取CP长度（优先用户指定，否则自动计算）
        if self.params.get("cp_length") is not None:
            self.cp_length = self.params.get("cp_length")
        else:
            # 自动计算CP长度：优先信道最大时延，否则用数据块比例
            chan_delays = self.params.get("chan_delays")
            if len(chan_delays) > 0:
                self.cp_length = max(chan_delays)  # 信道最大时延（码片数）
            else:
                self.cp_length = int(self.N * self.cp_ratio)
        
        # 校验CP长度合法性
        if self.cp_length < 0 or self.cp_length >= self.N:
            raise ValueError(f"CP长度({self.cp_length})需大于等于0且小于数据块长度({self.N})")

    def insert_cp(self, data):
        """
        为数据插入循环前缀
        :param data: 一维输入数据（任意长度）
        :return: 插入CP后的一维数据
        """
        # 补零使数据长度为N的整数倍
        if len(data) % self.N != 0:
            padding_length = self.N - len(data) % self.N
            data = np.pad(data, (0, padding_length), mode="constant")
        
        # 重塑为[N, num_blocks]的矩阵（每列一个数据块）
        data_blocks = data.reshape(-1, self.N).T  # 转置为列块结构
        
        # 插入CP：取每个块的最后cp_length个元素作为前缀，拼接在块前
        cp_blocks = np.concatenate([data_blocks[-self.cp_length:, :], data_blocks], axis=0)
        
        # 展平为一维信号（按列优先）
        data_with_cp = cp_blocks.T.flatten()
        
        return data_with_cp

    def remove_cp(self, data_with_cp):
        """
        从插入CP的数据中移除CP
        :param data_with_cp: 插入CP后的一维数据
        :return: 移除CP后的原始数据块
        """
        # 计算总块数和每个块的总长度（N+cp_length）
        total_block_length = self.N + self.cp_length
        num_blocks = len(data_with_cp) // total_block_length
        
        # 重塑为[total_block_length, num_blocks]的矩阵（每列一个带CP的块）
        data_blocks_with_cp = data_with_cp.reshape(num_blocks, total_block_length).T
        
        # 移除CP部分（保留后N个元素）
        data_blocks = data_blocks_with_cp[self.cp_length:, :]
        
        # 展平为一维数据并去除补零
        data_removed_cp = data_blocks.T.flatten()
        
        return data_removed_cp

# 测试
if __name__ == "__main__":
    from params.PHYParams import PHYParams
    params1 = PHYParams()
    cp_inserter1 = CPInserter(params1)
    data1 = np.random.randn(480) + 1j * np.random.randn(480)  # 1个数据块（512）
    data_with_cp1 = cp_inserter1.insert_cp(data1)
    data_removed_cp1 = cp_inserter1.remove_cp(data_with_cp1)
    print("="*50)
    print("测试用例1（自定义subframe_length=480，cp_length=32）：")
    print(f"原始数据长度：{len(data1)}")
    print(f"插入CP后长度：{len(data_with_cp1)}")  # 2*(128+32)=320
    print(f"移除CP后长度：{len(data_removed_cp1)}")
    print(f"CP长度：{cp_inserter1.cp_length}，数据块长度：{cp_inserter1.N}")