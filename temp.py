# 导入所需的库
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, BasicAer, execute
from qiskit.tools.visualization import plot_histogram, plot_bloch_multivector
import numpy as np

# 定义Alice和Bob的量子寄存器和经典寄存器
all_qubits_Alice = QuantumRegister(2)  # Alice的量子比特
all_qubits_Bob = QuantumRegister(1)    # Bob的量子比特
creg1_Alice = ClassicalRegister(1)     # Alice的第一个经典比特寄存器
creg2_Alice = ClassicalRegister(1)     # Alice的第二个经典比特寄存器

# 创建量子电路
mycircuit = QuantumCircuit(all_qubits_Alice, all_qubits_Bob, creg1_Alice, creg2_Alice)

# 初始化待传送的量子比特
mycircuit.initialize([np.sqrt(0.60), np.sqrt(0.40)], 0)  # 初始态 |0>

# 在电路中添加一个屏障以区分不同的操作步骤
mycircuit.barrier()

# 制备一对Bell基态粒子
mycircuit.h(1)   # 在第1个量子比特上施加Hadamard门
mycircuit.cx(1, 2)  # 控制NOT门，第1个比特为控制比特，第2个比特为目标比特

# 在电路中添加一个屏障
mycircuit.barrier()

# Alice测量，将结果放入两个经典寄存器中
mycircuit.cx(all_qubits_Alice[0], all_qubits_Alice[1])  # CNOT门，第1个比特为控制比特，第2个比特为目标比特
mycircuit.h(all_qubits_Alice[0])  # 在第0个量子比特上施加Hadamard门

# 在电路中添加一个屏障
mycircuit.barrier()

# 对Alice的量子比特进行测量，并将结果存储到经典寄存器中
mycircuit.measure(all_qubits_Alice[0], creg1_Alice)  # 对第0个量子比特测量，并将结果存储到第一个经典寄存器中
mycircuit.measure(all_qubits_Alice[1], creg2_Alice)  # 对第1个量子比特测量，并将结果存储到第二个经典寄存器中

# 在电路中添加一个屏障
mycircuit.barrier()

# Bob根据Alice的测量结果进行操作
mycircuit.x(all_qubits_Bob[0]).c_if(creg2_Alice, 1)  # 如果第二个经典寄存器的值为1，则对Bob的量子比特施加X门
mycircuit.z(all_qubits_Bob[0]).c_if(creg1_Alice, 1)  # 如果第一个经典寄存器的值为1，则对Bob的量子比特施加Z门

# 模拟电路，重复10000次以检验ZX四种情况的概率
backend = BasicAer.get_backend('qasm_simulator')
result = execute(mycircuit, backend, shots=10000).result()
state = result.get_counts(mycircuit)
plot_histogram(state)

# 模拟检验最终量子状态是否正确
backend = BasicAer.get_backend('statevector_simulator')
result = execute(mycircuit, backend).result()
state = result.get_statevector(mycircuit)
plot_bloch_multivector(state)

# 打印量子电路的状态向量
print(state)

# 绘制量子电路
mycircuit.draw('mpl')
plt.show()

