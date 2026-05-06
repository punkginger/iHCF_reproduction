import numpy as np
import random

class BaseScheduler:
    """
    Base class for scheduling algorithms.
    """
    def __init__(self, N, num_iters=1):
        self.N = N                  # Number of ports (N x N switch)
        self.num_iters = num_iters  # Number of iterations per time slot

    def schedule(self, voq_status):
        # Subclasses must implement this method
        raise NotImplementedError("Subclasses must implement the schedule method.")

class iHCF(BaseScheduler):
    """
    Implementation of the iHCF algorithm and its variants.
    Written in a C-like style for clarity.
    """
    def __init__(self, N, num_iters=1, max_cnt=None, tie_breaker='rr'):
        super().__init__(N, num_iters)
        
        # 1. 设置最大计数器 (None 表示无界 unbounded)
        self.max_cnt = max_cnt 
        
        # 2. 设置打破平局的策略 ('rr' 为轮询, 'random' 为随机)
        self.tie_breaker = tie_breaker
        
        self.voq_cntr = np.zeros((N, N), dtype=int)
        self.in_ptr = np.zeros(N, dtype=int)
        self.out_ptr = np.zeros(N, dtype=int)

    def schedule(self, voq_status):
        matched_in = []
        matched_out = []
        match_result = []
        
        for idx in range(self.N):
            matched_in.append(False)
            matched_out.append(False)
            match_result.append(-1)

        # ---------------------------------------------------------
        # 1. Update Counters 
        # ---------------------------------------------------------
        for i in range(self.N):
            for j in range(self.N):
                if voq_status[i][j] > 0:
                    self.voq_cntr[i][j] += 1
                    # 如果设定了最大值 (Saturating)，则进行限制
                    if self.max_cnt is not None:
                        if self.voq_cntr[i][j] > self.max_cnt:
                            self.voq_cntr[i][j] = self.max_cnt

        # ---------------------------------------------------------
        # 2. Iterative Matching Process
        # ---------------------------------------------------------
        for iteration in range(self.num_iters):
            
            # --- Phase 1: Request (Implicit) ---

            # --- Phase 2: Grant ---
            output_tied = [False] * self.N
            grants = []
            for idx in range(self.N):
                grants.append(-1)

            for j in range(self.N):
                if matched_out[j] == True: 
                    continue
                
                max_count = -1
                for i in range(self.N):
                    if matched_in[i] == False and voq_status[i][j] > 0:
                        if self.voq_cntr[i][j] > max_count:
                            max_count = self.voq_cntr[i][j]
                
                if max_count == -1:
                    continue
                
                candidates = []
                for i in range(self.N):
                    if matched_in[i] == False and voq_status[i][j] > 0:
                        if self.voq_cntr[i][j] == max_count:
                            candidates.append(i)
                
                # --- Tie-breaking Logic (Grant Phase) ---
                selected_i = -1
                if len(candidates) == 1:
                    selected_i = candidates[0]
                else:
                    if self.tie_breaker == 'random':
                        # 随机策略：从候选人中随机挑一个
                        rand_idx = random.randint(0, len(candidates) - 1)
                        selected_i = candidates[rand_idx]
                    else: # 'rr' 策略
                        # 轮询策略：使用 out_ptr
                        for offset in range(self.N):
                            idx = (self.out_ptr[j] + offset) % self.N
                            if idx in candidates:
                                selected_i = idx
                                break
                
                grants[j] = selected_i

            # --- Phase 3: Accept ---
            for i in range(self.N):
                if matched_in[i] == True: 
                    continue
                
                max_count = -1
                for j in range(self.N):
                    if grants[j] == i:
                        if self.voq_cntr[i][j] > max_count:
                            max_count = self.voq_cntr[i][j]
                
                if max_count == -1:
                    continue
                
                candidates = []
                for j in range(self.N):
                    if grants[j] == i:
                        if self.voq_cntr[i][j] == max_count:
                            candidates.append(j)
                
                # --- Tie-breaking Logic (Accept Phase) ---
                selected_j = -1
                input_tied = False
                if len(candidates) == 1:
                    selected_j = candidates[0]
                else:
                    if self.tie_breaker == 'random':
                        # 随机策略：从候选人中随机挑一个
                        rand_idx = random.randint(0, len(candidates) - 1)
                        selected_j = candidates[rand_idx]
                    else: # 'rr' 策略
                        # 轮询策略：使用 in_ptr
                        for offset in range(self.N):
                            idx = (self.in_ptr[i] + offset) % self.N
                            if idx in candidates:
                                selected_j = idx
                                break
                
                match_result[i] = selected_j
                matched_in[i] = True
                matched_out[selected_j] = True
                
                # --- State Reset & Pointer Updates ---
                self.voq_cntr[i][selected_j] = 0 
                
                # 指针更新仅在 'rr' 模式下有意义，为了逻辑严谨加上判断
                if self.tie_breaker == 'rr':
                    if output_tied[selected_j] == True:
                        self.out_ptr[selected_j] = (i + 1) % self.N
                    if input_tied == True:
                        self.in_ptr[i] = (selected_j + 1) % self.N

        return match_result

class iSLIP(BaseScheduler):
    """
    Implementation of the iSLIP algorithm.
    Written in a C-like style for clarity of underlying logic.
    """
    def __init__(self, N, num_iters=1):
        super().__init__(N, num_iters)
        
        # Core states: iSLIP ONLY needs pointers, no counters!
        self.in_ptr = np.zeros(N, dtype=int)  # Round-robin pointers for inputs (Accept phase)
        self.out_ptr = np.zeros(N, dtype=int) # Round-robin pointers for outputs (Grant phase)

    def schedule(self, voq_status):
        # Explicit initialization of status arrays (C-style)
        matched_in = []
        matched_out = []
        match_result = []
        
        for idx in range(self.N):
            matched_in.append(False)
            matched_out.append(False)
            match_result.append(-1)

        # ---------------------------------------------------------
        # Iterative Matching Process
        # ---------------------------------------------------------
        for iteration in range(self.num_iters):
            
            # --- Phase 1: Request ---
            # Requests are implicitly evaluated by checking voq_status[i][j] > 0

            # --- Phase 2: Grant ---
            grants = []
            for idx in range(self.N):
                grants.append(-1)

            for j in range(self.N):
                # If output j is already matched, skip it
                if matched_out[j] == True: 
                    continue
                
                # Output Arbiter makes a decision directly using its round-robin pointer
                # It scans from out_ptr[j] to find the FIRST eligible requesting input
                for offset in range(self.N):
                    # Calculate the input index to check based on current pointer and offset
                    i = (self.out_ptr[j] + offset) % self.N
                    
                    # Condition: Input i is unmatched AND has data for output j
                    if matched_in[i] == False and voq_status[i][j] > 0:
                        # Output j grants to input i
                        grants[j] = i
                        break # Stop searching as soon as the first request is found!

            # --- Phase 3: Accept ---
            for i in range(self.N):
                # If input i is already matched, skip it
                if matched_in[i] == True: 
                    continue
                
                # Input Arbiter makes a decision directly using its round-robin pointer
                # It scans from in_ptr[i] to find the FIRST output that granted to it
                for offset in range(self.N):
                    # Calculate the output index to check based on current pointer and offset
                    j = (self.in_ptr[i] + offset) % self.N
                    
                    # Condition: Output j actually granted to input i
                    if grants[j] == i:
                        # Establish the final match
                        match_result[i] = j
                        matched_in[i] = True
                        matched_out[j] = True
                        
                        # --- Pointer Updates (The secret to iSLIP's desynchronization) ---
                        # 1. Output pointer update: Moves ONLY IF the grant was accepted!
                        # (Because we are inside the Accept phase, we know it's accepted here)
                        self.out_ptr[j] = (i + 1) % self.N
                        
                        # 2. Input pointer update: Moves to one past the accepted output
                        self.in_ptr[i] = (j + 1) % self.N
                        
                        break # Stop searching as soon as the first grant is accepted!

        return match_result
    
class iOCF(BaseScheduler):
    """
    Implementation of the iOCF (Iterative Oldest Cell First) algorithm.
    Written in a C-like style for clarity.
    """
    def __init__(self, N, num_iters=1, tie_breaker='rr'):
        super().__init__(N, num_iters)
        
        self.tie_breaker = tie_breaker
        
        # iOCF 不需要自己维护计数器，只需维护打破平局用的指针
        self.in_ptr = np.zeros(N, dtype=int)
        self.out_ptr = np.zeros(N, dtype=int)

    def schedule(self, voq_status):
        matched_in = []
        matched_out = []
        match_result = []
        
        for idx in range(self.N):
            matched_in.append(False)
            matched_out.append(False)
            match_result.append(-1)

        # ---------------------------------------------------------
        # Iterative Matching Process
        # ---------------------------------------------------------
        for iteration in range(self.num_iters):
            
            # --- Phase 1: Request ---

            # --- Phase 2: Grant ---
            grants = []
            for idx in range(self.N):
                grants.append(-1)

            for j in range(self.N):
                if matched_out[j] == True: 
                    continue
                
                # Step 2.1: 找寻真实的最高权重 (Oldest Cell)
                max_weight = -1
                for i in range(self.N):
                    if matched_in[i] == False and voq_status[i][j] > 0:
                        # 直接使用 voq_status 里的真实等待时间作为比较权重
                        if voq_status[i][j] > max_weight:
                            max_weight = voq_status[i][j]
                
                if max_weight == -1:
                    continue
                
                # Step 2.2: 揪出所有达到最高权重的人
                candidates = []
                for i in range(self.N):
                    if matched_in[i] == False and voq_status[i][j] == max_weight:
                        candidates.append(i)
                
                # Step 2.3: 打破平局逻辑
                selected_i = -1
                if len(candidates) == 1:
                    selected_i = candidates[0]
                else:
                    if self.tie_breaker == 'random':
                        rand_idx = random.randint(0, len(candidates) - 1)
                        selected_i = candidates[rand_idx]
                    else: # 'rr' 策略
                        for offset in range(self.N):
                            idx = (self.out_ptr[j] + offset) % self.N
                            if idx in candidates:
                                selected_i = idx
                                break
                
                grants[j] = selected_i

            # --- Phase 3: Accept ---
            for i in range(self.N):
                if matched_in[i] == True: 
                    continue
                
                # Step 3.1: 找寻真实的最高权重
                max_weight = -1
                for j in range(self.N):
                    if grants[j] == i:
                        if voq_status[i][j] > max_weight:
                            max_weight = voq_status[i][j]
                
                if max_weight == -1:
                    continue
                
                # Step 3.2: 揪出所有达到最高权重的授权方
                candidates = []
                for j in range(self.N):
                    if grants[j] == i and voq_status[i][j] == max_weight:
                        candidates.append(j)
                
                # Step 3.3: 打破平局逻辑
                selected_j = -1
                if len(candidates) == 1:
                    selected_j = candidates[0]
                else:
                    if self.tie_breaker == 'random':
                        rand_idx = random.randint(0, len(candidates) - 1)
                        selected_j = candidates[rand_idx]
                    else: # 'rr' 策略
                        for offset in range(self.N):
                            idx = (self.in_ptr[i] + offset) % self.N
                            if idx in candidates:
                                selected_j = idx
                                break
                
                match_result[i] = selected_j
                matched_in[i] = True
                matched_out[selected_j] = True
                
                # --- Pointer Updates ---
                # 注意：iOCF 不需要重置计数器（因为它根本没有内部计数器）
                if self.tie_breaker == 'rr':
                    self.out_ptr[selected_j] = (i + 1) % self.N
                    if len(candidates) > 1:
                        self.in_ptr[i] = (selected_j + 1) % self.N

        return match_result
    
class iLPF(BaseScheduler):
    """
    Implementation of the iLPF (Iterative Longest Port First) algorithm.
    Written in a C-like style for clarity.
    """
    def __init__(self, N, num_iters=1, tie_breaker='rr'):
        super().__init__(N, num_iters)
        
        self.tie_breaker = tie_breaker
        self.in_ptr = np.zeros(N, dtype=int)
        self.out_ptr = np.zeros(N, dtype=int)

    def schedule(self, voq_status):
        matched_in = []
        matched_out = []
        match_result = []
        
        for idx in range(self.N):
            matched_in.append(False)
            matched_out.append(False)
            match_result.append(-1)

        # ---------------------------------------------------------
        # Pre-calculation: 计算每个端口的总积压量 (Port Occupancy)
        # ---------------------------------------------------------
        input_occupancy = []   # R_i (Row sum)
        output_occupancy = []  # C_j (Column sum)
        
        for idx in range(self.N):
            input_occupancy.append(0)
            output_occupancy.append(0)
            
        for i in range(self.N):
            for j in range(self.N):
                input_occupancy[i] += voq_status[i][j]
                output_occupancy[j] += voq_status[i][j]

        # ---------------------------------------------------------
        # Iterative Matching Process
        # ---------------------------------------------------------
        for iteration in range(self.num_iters):
            
            # --- Phase 1: Request ---

            # --- Phase 2: Grant ---
            grants = []
            for idx in range(self.N):
                grants.append(-1)

            for j in range(self.N):
                if matched_out[j] == True: 
                    continue
                
                # Step 2.1: 找寻最大的端口权重 (Max Weight = R_i + C_j)
                max_weight = -1
                for i in range(self.N):
                    # 前提条件：队列里必须有数据 (voq_status[i][j] > 0)
                    if matched_in[i] == False and voq_status[i][j] > 0:
                        weight = input_occupancy[i] + output_occupancy[j]
                        if weight > max_weight:
                            max_weight = weight
                
                if max_weight == -1:
                    continue
                
                # Step 2.2: 找出所有达到最大权重的人
                candidates = []
                for i in range(self.N):
                    if matched_in[i] == False and voq_status[i][j] > 0:
                        weight = input_occupancy[i] + output_occupancy[j]
                        if weight == max_weight:
                            candidates.append(i)
                
                # Step 2.3: 打破平局逻辑
                selected_i = -1
                if len(candidates) == 1:
                    selected_i = candidates[0]
                else:
                    if self.tie_breaker == 'random':
                        rand_idx = random.randint(0, len(candidates) - 1)
                        selected_i = candidates[rand_idx]
                    else: # 'rr' 策略
                        for offset in range(self.N):
                            idx = (self.out_ptr[j] + offset) % self.N
                            if idx in candidates:
                                selected_i = idx
                                break
                
                grants[j] = selected_i

            # --- Phase 3: Accept ---
            for i in range(self.N):
                if matched_in[i] == True: 
                    continue
                
                # Step 3.1: 找寻最大的端口权重
                max_weight = -1
                for j in range(self.N):
                    if grants[j] == i:
                        weight = input_occupancy[i] + output_occupancy[j]
                        if weight > max_weight:
                            max_weight = weight
                
                if max_weight == -1:
                    continue
                
                # Step 3.2: 找出所有达到最大权重的授权方
                candidates = []
                for j in range(self.N):
                    if grants[j] == i:
                        weight = input_occupancy[i] + output_occupancy[j]
                        if weight == max_weight:
                            candidates.append(j)
                
                # Step 3.3: 打破平局逻辑
                selected_j = -1
                if len(candidates) == 1:
                    selected_j = candidates[0]
                else:
                    if self.tie_breaker == 'random':
                        rand_idx = random.randint(0, len(candidates) - 1)
                        selected_j = candidates[rand_idx]
                    else: # 'rr' 策略
                        for offset in range(self.N):
                            idx = (self.in_ptr[i] + offset) % self.N
                            if idx in candidates:
                                selected_j = idx
                                break
                
                match_result[i] = selected_j
                matched_in[i] = True
                matched_out[selected_j] = True
                
                # --- Pointer Updates ---
                if self.tie_breaker == 'rr':
                    self.out_ptr[selected_j] = (i + 1) % self.N
                    if len(candidates) > 1:
                        self.in_ptr[i] = (selected_j + 1) % self.N

        return match_result
    
