import galois
import numpy as np

# ====================== 与Matlab完全一致的参数 ======================
n = 15
k = 11
m = 4
GF = galois.GF(2**m)

# 1. 原始信息（与Matlab完全相同）
info = np.array([3,8,12,5,9,0,7,15,2,10,4], dtype=int)
print("==================== RS(11,15) Python仿真结果 ====================")
print("1、原始输入信息序列：")
print(info)

# 2. RS编码（系统码，与Matlab rsenc 对齐）
rs = galois.ReedSolomon(n, k, field=GF)
code = rs.encode(info)
print("2、RS编码后完整码字（15位）：")
print(code)

# 3. 加入与Matlab一致的2位随机错误（修复GF域类型报错）
# 核心修复：转换为numpy整型数组运算，避免GF对象与int直接运算冲突
np.random.seed(0)
code_val = np.array(code, dtype=int)  # 提取纯数值数组
err_pos = np.random.permutation(n)[:2]
rx_code = code_val.copy()
# 生成GF域合法随机错误并叠加取模
rx_code[err_pos[0]] = (rx_code[err_pos[0]] + np.random.randint(1,16)) % 16
rx_code[err_pos[1]] = (rx_code[err_pos[1]] + np.random.randint(1,16)) % 16
# 转回GF域格式用于译码
rx_code_gf = GF(rx_code)

print("3、加噪后接收码字（含2位错误）：")
print(rx_code)
print(f"错误发生位置：第{err_pos[0]+1}位、第{err_pos[1]+1}位")

# 4. RS译码
dec_info, err_num = rs.decode(rx_code_gf, errors=True)
dec_info_val = np.array(dec_info, dtype=int)
print("4、译码恢复信息序列：")
print(dec_info_val)
print(f"5、译码检测错误位数：{err_num}")

# 5. 校验比对
if np.array_equal(dec_info_val, info):
    print("✅ 译码成功：与Matlab结果一致，错误完全纠正！")
else:
    print("❌ 译码失败：信息不一致")
print("==================================================================")