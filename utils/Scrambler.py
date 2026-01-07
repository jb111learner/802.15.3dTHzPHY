# utils/Scrambler.py
import numpy as np

class Scrambler:
    """
    扰码器/解扰码器：基于线性反馈移位寄存器（LFSR）实现
    遵循802.15.3d协议的扰码规则（也可自定义多项式）
    多项式：x^7 + x^4 + 1（经典7阶扰码器，避免长0/长1）
    """
    def __init__(self, params):
        self.params = params
        # 扰码器初始状态：强制转为uint8的一维数组，避免类型/维度问题
        self.init_state = np.array(self.params.get("scrambler_init"), dtype=np.uint8)
        # 校验初始状态长度（7阶LFSR必须是7位）
        if len(self.init_state) != 7:
            raise ValueError(f"初始状态必须是7位，当前长度：{len(self.init_state)}")
        
        self.poly = [7,4]  # 反馈抽头（对应x^7和x^4）
        self.state = self.init_state.copy()  # 当前移位寄存器状态

    def _reset(self):
        """重置移位寄存器状态"""
        self.state = self.init_state.copy()

    def scramble(self, data_bits):
        """
        扰码：将输入比特流与LFSR生成的伪随机序列逐位异或
        :param data_bits: 输入二进制比特流（np.array，0/1）
        :return: 扰码后的比特流
        """
        self._reset()
        # 强制输入为uint8一维数组，避免形状问题
        data_bits = np.asarray(data_bits, dtype=np.uint8).flatten()
        scrambled_bits = np.zeros_like(data_bits)
        
        for i in range(len(data_bits)):
            # 1. 计算反馈位（抽头位异或）
            feedback = self.state[self.poly[0]-1] ^ self.state[self.poly[1]-1]
            # 2. 输出位 = 输入位 ^ 移位寄存器最后一位
            scrambled_bits[i] = data_bits[i] ^ self.state[-1]
            # 3. 移位寄存器右移：修复核心赋值逻辑，避免形状不匹配
            # 方式1：先创建临时数组，再整体赋值（最稳妥）
            temp_state = np.roll(self.state, shift=1)  # 右移1位
            temp_state[0] = feedback  # 反馈位填入第一位
            self.state = temp_state
            # （替代方式2：逐位赋值，适合新手理解）
            # for j in range(len(self.state)-1, 0, -1):
            #     self.state[j] = self.state[j-1]
            # self.state[0] = feedback
        
        return scrambled_bits

    def descramble(self, scrambled_bits):
        """
        解扰码：与扰码使用完全相同的LFSR流程（自同步特性）
        :param scrambled_bits: 扰码后的比特流
        :return: 解扰后的原始比特流
        """
        return self.scramble(scrambled_bits)  # 扰码和解扰算法完全一致
    

def test_scrambler():
    """测试扰码器/解扰码器功能"""
    # 1. 配置扰码器参数（初始状态为7位非零值，符合802.15.3d协议）
    scrambler_params = {
        "scrambler_init": np.array([1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)  # 7位初始状态
    }
    
    # 2. 创建扰码器实例
    scrambler = Scrambler(scrambler_params)
    
    # 3. 准备测试用例（覆盖不同场景）
    test_cases = [
        {
            "name": "随机比特流",
            "data": np.random.randint(0, 2, size=100, dtype=np.uint8)  # 100位随机比特
        },
        {
            "name": "全0比特流",
            "data": np.zeros(50, dtype=np.uint8)  # 50位全0（验证避免长0）
        },
        {
            "name": "全1比特流",
            "data": np.ones(30, dtype=np.uint8)  # 30位全1（验证避免长1）
        },
        {
            "name": "短比特流（7位）",
            "data": np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.uint8)  # 最短测试用例
        }
    ]
    
    # 4. 执行测试
    print("=" * 60)
    print("开始测试扰码器/解扰码器功能")
    print("=" * 60)
    
    for case in test_cases:
        print(f"\n【测试用例：{case['name']}】")
        original_data = case["data"]
        
        # 扰码
        scrambled_data = scrambler.scramble(original_data)
        # 解扰码
        descrambled_data = scrambler.descramble(scrambled_data)
        
        # 验证结果
        is_correct = np.array_equal(original_data, descrambled_data)
        
        # 输出关键信息
        print(f"原始数据前10位: {original_data[:10]}")
        print(f"扰码后前10位: {scrambled_data[:10]}")
        print(f"解扰后前10位: {descrambled_data[:10]}")
        print(f"验证结果: {'✅ 正确' if is_correct else '❌ 错误'}")
        
        # 额外验证：扰码后的数据不应与原始数据完全相同（除了空数据）
        if len(original_data) > 0:
            is_scrambled = not np.array_equal(original_data, scrambled_data)
            print(f"扰码有效性: {'✅ 有效' if is_scrambled else '❌ 无效（扰码无变化）'}")

if __name__ == "__main__":
    # 设置随机种子，保证测试结果可复现
    np.random.seed(42)
    # 执行测试
    test_scrambler()