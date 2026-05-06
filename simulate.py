import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

# 导入你写好的算法 (确保 scheduler.py 在同一目录下)
from scheduler import iHCF, iSLIP, iOCF, iLPF

def run_latency_simulation(scheduler, algo_name, N, load, time_slots, traffic_type):
    """
    统一仿真环境，支持 3 种不同的流量模型
    """
    voqs = [[deque() for _ in range(N)] for _ in range(N)]
    
    total_delay = 0
    total_served = 0
    warmup_time = int(time_slots * 0.1)


    # 在 run_latency_simulation 函数内部，循环开始前初始化状态
    # input_states: 记录每个输入端口当前是否处于 "On (发包)" 状态
    # input_targets: 记录处于 "On" 状态时，包要发往哪个输出端口
    burst_active = [False] * N
    burst_dest = [-1] * N

    # 设定突发性参数 (可以根据需要调整)
    # p_on: 从 Off 转为 On 的概率 (决定了新突发的产生)
    # p_off: 从 On 转为 Off 的概率 (决定了突发的平均长度，1/p_off 越小，突发越长)
    p_off = 0.2  # 意味着平均每个突发长度为 5 个包

    for t in range(time_slots):
        # ---------------------------------------------------------
        # 1. 流量生成 (根据传入的 traffic_type 决定)
        # ---------------------------------------------------------
        for i in range(N):
            if traffic_type == 'uniform':
                if random.random() < load:
                    dest_j = random.randint(0, N - 1)
                    voqs[i][dest_j].append(t)
                    
            elif traffic_type == 'non_uniform':
                if random.random() < load:
                    # 示例：2/3 概率发给 i，1/3 发给 i+1
                    # (这也是常见使 iSLIP 劣化的非均匀模式)
                    if random.random() < 2/3:
                        dest_j = i
                    else:
                        dest_j = (i + 1) % N
                    voqs[i][dest_j].append(t)
                        
            elif traffic_type == 'hot_spot':
                # Admissible Hot-spot Traffic
                if random.random() < load:
                    weights = [2.0 if j == 0 else 1.0 for j in range(N)]
                    dest_j = random.choices(range(N), weights=weights, k=1)[0]
                    voqs[i][dest_j].append(t)
                
        # ---------------------------------------------------------
        # 2. 提取 voq_status 
        # ---------------------------------------------------------
        voq_status = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                if len(voqs[i][j]) > 0:
                    if algo_name == 'iOCF':
                        arrival_time = voqs[i][j][0]
                        voq_status[i][j] = (t - arrival_time) + 1 
                    else:
                        voq_status[i][j] = len(voqs[i][j])
                        
        # ---------------------------------------------------------
        # 3. 算法调度 & 4. 统计延迟
        # ---------------------------------------------------------
        matches = scheduler.schedule(voq_status)
        
        for i in range(N):
            j = matches[i]
            if j != -1:
                arrival_time = voqs[i][j].popleft()
                if t >= warmup_time:
                    delay = t - arrival_time
                    total_delay += delay
                    total_served += 1
                    
    if total_served > 0:
        return total_delay / total_served
    else:
        return 0.0

# =====================================================================
# 辅助绘图函数
# =====================================================================
def plot_and_save(loads, data_dict, title, filename):
    plt.figure(figsize=(7, 6))
    markers = ['s', 'o', 'v', 'x', 'd', '^']
    colors = ['r', 'b', 'g', 'm', 'c', 'k']
    
    for idx, (label, delays) in enumerate(data_dict.items()):
        plt.plot(loads, delays, marker=markers[idx%len(markers)], color=colors[idx%len(colors)], linestyle='-', label=label)
        
    plt.yscale('log')
    plt.xlabel('Load')
    plt.ylabel('Average Latency (Time Slots)')
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"--> 已保存图表: {filename}")

if __name__ == "__main__":
    N_ports = 16
    time_slots = 10000 
    num_iterations = 4 
    loads = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    
    hcf_max_cnt = N_ports - 1 
    
    print(f"全局设定: N={N_ports}, max_cnt={hcf_max_cnt}, Iterations={num_iterations}")
    # =====================================================================
    # 实验 1: Uniform Traffic 下 iHCF 三种变体的区别
    # =====================================================================
    print("\n[开始] 实验 1: iHCF 变体对比 (Uniform Traffic)")
    res_exp1 = {'iHCF-unbounded': [], 'iHCF-random': [], 'iHCF-RR (Standard)': []}
    
    for load in loads:
        s_unbound = iHCF(N=N_ports, num_iters=num_iterations, max_cnt=None, tie_breaker='rr')
        s_rand    = iHCF(N=N_ports, num_iters=num_iterations, max_cnt=hcf_max_cnt, tie_breaker='random')
        s_rr      = iHCF(N=N_ports, num_iters=num_iterations, max_cnt=hcf_max_cnt, tie_breaker='rr')
        
        res_exp1['iHCF-unbounded'].append(run_latency_simulation(s_unbound, 'iHCF', N_ports, load, time_slots, 'uniform'))
        res_exp1['iHCF-random'].append(run_latency_simulation(s_rand, 'iHCF', N_ports, load, time_slots, 'uniform'))
        res_exp1['iHCF-RR (Standard)'].append(run_latency_simulation(s_rr, 'iHCF', N_ports, load, time_slots, 'uniform'))
        
    plot_and_save(loads, res_exp1, 'Exp 1: iHCF Variants (Uniform Traffic)', 'exp1_ihcf_variants_uniform.png')

    # =====================================================================
    # 实验 2: Uniform Traffic 下 几种算法的区别
    # =====================================================================
    print("\n[开始] 实验 2: 核心算法对比 (Uniform Traffic)")
    res_exp2 = {'iSLIP': [], 'iHCF': [], 'iOCF': [], 'iLPF': []}
    
    for load in loads:
        res_exp2['iSLIP'].append(run_latency_simulation(iSLIP(N=N_ports, num_iters=num_iterations), 'iSLIP', N_ports, load, time_slots, 'uniform'))
        res_exp2['iHCF'].append(run_latency_simulation(iHCF(N=N_ports, num_iters=num_iterations, max_cnt=hcf_max_cnt, tie_breaker='rr'), 'iHCF', N_ports, load, time_slots, 'uniform'))
        res_exp2['iOCF'].append(run_latency_simulation(iOCF(N=N_ports, num_iters=num_iterations, tie_breaker='rr'), 'iOCF', N_ports, load, time_slots, 'uniform'))
        res_exp2['iLPF'].append(run_latency_simulation(iLPF(N=N_ports, num_iters=num_iterations, tie_breaker='rr'), 'iLPF', N_ports, load, time_slots, 'uniform'))
        
    plot_and_save(loads, res_exp2, 'Exp 2: Algorithm Comparison (Uniform Traffic)', 'exp2_comparison_uniform.png')

    # =====================================================================
    # 实验 3: Non-uniform Traffic 下 几种算法的区别
    # =====================================================================
    print("\n[开始] 实验 3: 核心算法对比 (Non-uniform Traffic)")
    res_exp3 = {'iSLIP': [], 'iHCF': [], 'iOCF': [], 'iLPF': []}
    
    for load in loads:
        res_exp3['iSLIP'].append(run_latency_simulation(iSLIP(N=N_ports, num_iters=num_iterations), 'iSLIP', N_ports, load, time_slots, 'non_uniform'))
        res_exp3['iHCF'].append(run_latency_simulation(iHCF(N=N_ports, num_iters=num_iterations, max_cnt=hcf_max_cnt, tie_breaker='rr'), 'iHCF', N_ports, load, time_slots, 'non_uniform'))
        res_exp3['iOCF'].append(run_latency_simulation(iOCF(N=N_ports, num_iters=num_iterations, tie_breaker='rr'), 'iOCF', N_ports, load, time_slots, 'non_uniform'))
        res_exp3['iLPF'].append(run_latency_simulation(iLPF(N=N_ports, num_iters=num_iterations, tie_breaker='rr'), 'iLPF', N_ports, load, time_slots, 'non_uniform'))
        
    plot_and_save(loads, res_exp3, 'Exp 3: Algorithm Comparison (Non-uniform Traffic)', 'exp3_comparison_nonuniform.png')

    # =====================================================================
    # 实验 4: Hot-spot Traffic 下 几种算法的区别
    # =====================================================================
    print("\n[开始] 实验 4: 核心算法对比 (Hot-spot Traffic)")
    res_exp4 = {'iSLIP': [], 'iHCF': [], 'iOCF': [], 'iLPF': []}
    
    # 热点流量极易引发拥塞，因此可以适当缩短测试范围，或直接跑完看哪里爆炸
    for load in loads:
        res_exp4['iSLIP'].append(run_latency_simulation(iSLIP(N=N_ports, num_iters=num_iterations), 'iSLIP', N_ports, load, time_slots, 'hot_spot'))
        res_exp4['iHCF'].append(run_latency_simulation(iHCF(N=N_ports, num_iters=num_iterations, max_cnt=hcf_max_cnt, tie_breaker='rr'), 'iHCF', N_ports, load, time_slots, 'hot_spot'))
        res_exp4['iOCF'].append(run_latency_simulation(iOCF(N=N_ports, num_iters=num_iterations, tie_breaker='rr'), 'iOCF', N_ports, load, time_slots, 'hot_spot'))
        res_exp4['iLPF'].append(run_latency_simulation(iLPF(N=N_ports, num_iters=num_iterations, tie_breaker='rr'), 'iLPF', N_ports, load, time_slots, 'hot_spot'))
        
    plot_and_save(loads, res_exp4, 'Exp 4: Algorithm Comparison (Hot-spot Traffic)', 'exp4_comparison_hotspot.png')