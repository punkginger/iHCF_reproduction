import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from scheduler import iHCF, iSLIP, iOCF, iLPF

def run_latency_simulation(scheduler, algo_name, N, load, time_slots, traffic_type):
    # N*N deque matrix. Each entry voqs[i][j] is a queue of arrival timestamps for cells at input i destined for output j.
    voqs = [[deque() for _ in range(N)] for _ in range(N)]
    
    total_delay = 0
    total_served = 0
    warmup_time = int(time_slots * 0.1) # 10% duration dismissed for warm-up


    for t in range(time_slots):
        # 1. traffic generation
        for i in range(N):
            if traffic_type == 'uniform':
                if random.random() < load:
                    dest_j = random.randint(0, N - 1)
                    voqs[i][dest_j].append(t)
                    
            elif traffic_type == 'non_uniform':
                if random.random() < load:
                    # 2/3 probability to i，1/3 to i+1
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
                
        # 2. get voq_status 
        voq_status = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                if len(voqs[i][j]) > 0:
                    if algo_name == 'iOCF':
                        arrival_time = voqs[i][j][0]
                        voq_status[i][j] = (t - arrival_time) + 1 
                    else:
                        voq_status[i][j] = len(voqs[i][j])
                        
        # 3. call scheduler to get matches
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


# plotting function
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
    print(f"--> picture saved: {filename}")

if __name__ == "__main__":
    N_ports = 4
    time_slots = 10000 
    num_iterations = 1
    loads = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    
    hcf_max_cnt = N_ports - 1 
    
    print(f"N={N_ports}, max_cnt={hcf_max_cnt}, Iterations={num_iterations}")
    # exp1: 3 variants of iHCF (Uniform Traffic)
    print("\n[start] exp 1: 3 variants of iHCF (Uniform Traffic)")
    res_exp1 = {'iHCF-unbounded': [], 'iHCF-random': [], 'iHCF-RR (Standard)': []}
    
    for load in loads:
        s_unbound = iHCF(N=N_ports, num_iters=num_iterations, max_cnt=None, tie_breaker='rr')
        s_rand    = iHCF(N=N_ports, num_iters=num_iterations, max_cnt=hcf_max_cnt, tie_breaker='random')
        s_rr      = iHCF(N=N_ports, num_iters=num_iterations, max_cnt=hcf_max_cnt, tie_breaker='rr')
        
        res_exp1['iHCF-unbounded'].append(run_latency_simulation(s_unbound, 'iHCF', N_ports, load, time_slots, 'uniform'))
        res_exp1['iHCF-random'].append(run_latency_simulation(s_rand, 'iHCF', N_ports, load, time_slots, 'uniform'))
        res_exp1['iHCF-RR (Standard)'].append(run_latency_simulation(s_rr, 'iHCF', N_ports, load, time_slots, 'uniform'))
        
    plot_and_save(loads, res_exp1, 'Exp 1: iHCF Variants (Uniform Traffic)', 'exp1_ihcf_variants_uniform.png')

    # exp2: iSLIP, iHCF, iOCF, iLPF (Uniform Traffic)
    print("\n[start] exp 2: Algorithm Comparison (Uniform Traffic)")
    res_exp2 = {'iSLIP': [], 'iHCF': [], 'iOCF': [], 'iLPF': []}
    
    for load in loads:
        res_exp2['iSLIP'].append(run_latency_simulation(iSLIP(N=N_ports, num_iters=num_iterations), 'iSLIP', N_ports, load, time_slots, 'uniform'))
        res_exp2['iHCF'].append(run_latency_simulation(iHCF(N=N_ports, num_iters=num_iterations, max_cnt=hcf_max_cnt, tie_breaker='rr'), 'iHCF', N_ports, load, time_slots, 'uniform'))
        res_exp2['iOCF'].append(run_latency_simulation(iOCF(N=N_ports, num_iters=num_iterations, tie_breaker='rr'), 'iOCF', N_ports, load, time_slots, 'uniform'))
        res_exp2['iLPF'].append(run_latency_simulation(iLPF(N=N_ports, num_iters=num_iterations, tie_breaker='rr'), 'iLPF', N_ports, load, time_slots, 'uniform'))
        
    plot_and_save(loads, res_exp2, 'Exp 2: Algorithm Comparison (Uniform Traffic)', 'exp2_comparison_uniform.png')

    # exp3: Non-uniform Traffic 
    print("\n[start] exp 3: Algorithm Comparison (Non-uniform Traffic)")
    res_exp3 = {'iSLIP': [], 'iHCF': [], 'iOCF': [], 'iLPF': []}
    
    for load in loads:
        res_exp3['iSLIP'].append(run_latency_simulation(iSLIP(N=N_ports, num_iters=num_iterations), 'iSLIP', N_ports, load, time_slots, 'non_uniform'))
        res_exp3['iHCF'].append(run_latency_simulation(iHCF(N=N_ports, num_iters=num_iterations, max_cnt=hcf_max_cnt, tie_breaker='rr'), 'iHCF', N_ports, load, time_slots, 'non_uniform'))
        res_exp3['iOCF'].append(run_latency_simulation(iOCF(N=N_ports, num_iters=num_iterations, tie_breaker='rr'), 'iOCF', N_ports, load, time_slots, 'non_uniform'))
        res_exp3['iLPF'].append(run_latency_simulation(iLPF(N=N_ports, num_iters=num_iterations, tie_breaker='rr'), 'iLPF', N_ports, load, time_slots, 'non_uniform'))
        
    plot_and_save(loads, res_exp3, 'Exp 3: Algorithm Comparison (Non-uniform Traffic)', 'exp3_comparison_nonuniform.png')


    # exp4: Hot-spot Traffic 
    print("\n[start] exp 4: Algorithm Comparison (Hot-spot Traffic)")
    res_exp4 = {'iSLIP': [], 'iHCF': [], 'iOCF': [], 'iLPF': []}
    
    for load in loads:
        res_exp4['iSLIP'].append(run_latency_simulation(iSLIP(N=N_ports, num_iters=num_iterations), 'iSLIP', N_ports, load, time_slots, 'hot_spot'))
        res_exp4['iHCF'].append(run_latency_simulation(iHCF(N=N_ports, num_iters=num_iterations, max_cnt=hcf_max_cnt, tie_breaker='rr'), 'iHCF', N_ports, load, time_slots, 'hot_spot'))
        res_exp4['iOCF'].append(run_latency_simulation(iOCF(N=N_ports, num_iters=num_iterations, tie_breaker='rr'), 'iOCF', N_ports, load, time_slots, 'hot_spot'))
        res_exp4['iLPF'].append(run_latency_simulation(iLPF(N=N_ports, num_iters=num_iterations, tie_breaker='rr'), 'iLPF', N_ports, load, time_slots, 'hot_spot'))
        
    plot_and_save(loads, res_exp4, 'Exp 4: Algorithm Comparison (Hot-spot Traffic)', 'exp4_comparison_hotspot.png')