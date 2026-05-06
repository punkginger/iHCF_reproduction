import random
import scheduler
import numpy as np
def run_simulation(scheduler, N, time_slots, arrival_rate):
    """
    A simple simulation environment to test the throughput of scheduling algorithms.
    """
    # Physical VOQ matrix (stores actual queue lengths)
    voq = np.zeros((N, N), dtype=int)
    
    total_arrived = 0
    total_served = 0

    for t in range(time_slots):
        # 1. Traffic Generation (Uniform Traffic)
        for i in range(N):
            # Simulate Bernoulli arrival process for each input port
            if random.random() < arrival_rate:
                # Randomly select a destination port (Uniform distribution)
                dest_j = random.randint(0, N - 1)
                voq[i][dest_j] += 1
                total_arrived += 1
                
        # 2. Extract VOQ status matrix (1 if queue is not empty, 0 otherwise)
        # Explicit C-style extraction
        voq_status = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                if voq[i][j] > 0:
                    voq_status[i][j] = 1
                else:
                    voq_status[i][j] = 0
        
        # 3. Call the scheduling algorithm
        matches = scheduler.schedule(voq_status)
        
        # 4. Update physical queues and statistics based on match results
        for i in range(N):
            j = matches[i]
            if j != -1: # If input i successfully matched with output j
                voq[i][j] -= 1
                total_served += 1
                
    # Calculate final throughput
    if total_arrived > 0:
        throughput = total_served / total_arrived
    else:
        throughput = 0.0
        
    return throughput

# ====== Test Example ======
if __name__ == "__main__":
    N_ports = 16          # 16x16 switch
    traffic_load = 0.8    # 80% input load
    
    # Instantiate the iHCF scheduler (1 iteration)
    my_ihcf_scheduler = scheduler.iHCF(N=N_ports, num_iters=1)
    
    # Run simulation
    throughput_result = run_simulation(my_ihcf_scheduler, N_ports, time_slots=10000, arrival_rate=traffic_load)
    
    print(f"iHCF Algorithm Throughput at Load {traffic_load}: {throughput_result:.4f}")