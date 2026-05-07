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
        raise NotImplementedError("All scheduler must implement the schedule method.")

class iHCF(BaseScheduler):
    def __init__(self, N, num_iters=1, max_cnt=None, tie_breaker='rr'):
        super().__init__(N, num_iters)

        """
        N: Number of ports (N x N switch)
        num_iters: Number of iterations for the scheduling process
        max_cnt: Maximum value for the internal counters 
        tie_breaker: Strategy for breaking ties 
        """
        # None = unbounded
        self.max_cnt = max_cnt 
        
        # rr = round-robin, random = random selection
        self.tie_breaker = tie_breaker
        
        """
        voq_cntr: NxN matrix tracking the waiting time of cells in each voq to determine priority. 
        in_ptr: Round-robin pointer for each input port, used for tie-breaking in the Accept phase.
        out_ptr: Round-robin pointer for each output port, used for tie-breaking in the Grant phase.
        """
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

        # 1. Update Counters 
        for i in range(self.N):
            for j in range(self.N):
                if voq_status[i][j] > 0:
                    self.voq_cntr[i][j] += 1
                    if self.max_cnt is not None:
                        if self.voq_cntr[i][j] > self.max_cnt:
                            self.voq_cntr[i][j] = self.max_cnt


        # 2. Iterative Matching Process
        for iteration in range(self.num_iters):
            
            # Phase 1: Request 
            # the Request phase is implicit in the sense that the presence of a request is determined by checking if voq_status[i][j] > 0.

            # Phase 2: Grant
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
                
                # Tie-breaking (Grant Phase)
                selected_i = -1
                if len(candidates) == 1:
                    selected_i = candidates[0]
                else:
                    if self.tie_breaker == 'random':
                        # randomly select one from candidates
                        selected_i = random.choice(candidates)
                    else: # round robin
                        # since it's grant phase, use out_ptr
                        for offset in range(self.N):
                            idx = (self.out_ptr[j] + offset) % self.N
                            if idx in candidates:
                                selected_i = idx
                                break
                
                grants[j] = selected_i

            # Phase 3: Accept
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
                
                # Tie-breaking (Accept Phase)
                selected_j = -1
                input_tied = False
                if len(candidates) == 1:
                    selected_j = candidates[0]
                else:
                    input_tied = True
                    if self.tie_breaker == 'random':
                        # randomly select one from candidates
                        selected_j = random.choice(candidates)
                    else: # 'rr' strategy
                        # round-robin strategy: use in_ptr
                        for offset in range(self.N):
                            idx = (self.in_ptr[i] + offset) % self.N
                            if idx in candidates:
                                selected_j = idx
                                break
                
                match_result[i] = selected_j
                matched_in[i] = True
                matched_out[selected_j] = True
                
                # State Reset & Pointer Updates 
                self.voq_cntr[i][selected_j] = 0 
                
                # Pointer updates are only meaningful in 'rr' mode, for logical consistency
                if self.tie_breaker == 'rr':
                    self.out_ptr[selected_j] = (i + 1) % self.N # unconditional
                if input_tied: # only if a tie in the accept phase
                    self.in_ptr[i] = (selected_j + 1) % self.N

        return match_result

class iSLIP(BaseScheduler):
    def __init__(self, N, num_iters=1):
        super().__init__(N, num_iters)
        
        """
        in_ptr: Round-robin pointer for each input port, used in the Accept phase.
        out_ptr: Round-robin pointer for each output port, used  in the Grant phase.
        """
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

        # Iterative Matching Process
        for iteration in range(self.num_iters):
            
            # Phase 1: Request 
            # Requests are implicitly evaluated by checking voq_status[i][j] > 0

            # Phase 2: Grant 
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

            # Phase 3: Accept 
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
                        
                        # Pointer Updates 
                        # 1. Output pointer update: Moves only if the grant was accepted
                        self.out_ptr[j] = (i + 1) % self.N
                        
                        # 2. Input pointer update: Moves to one past the accepted output
                        self.in_ptr[i] = (j + 1) % self.N
                        
                        break # Stop searching as soon as the first grant is accepted

        return match_result
    
class iOCF(BaseScheduler):
    def __init__(self, N, num_iters=1, tie_breaker='rr'):
        super().__init__(N, num_iters)
        
        self.tie_breaker = tie_breaker
        
        """
        in_ptr: Round-robin pointer for each input port, used in the Accept phase.
        out_ptr: Round-robin pointer for each output port, used  in the Grant phase.
        """
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

        # Iterative Matching Process
        for iteration in range(self.num_iters):
            
            # Phase 1: Request

            # Phase 2: Grant
            grants = []
            for idx in range(self.N):
                grants.append(-1)

            for j in range(self.N):
                if matched_out[j] == True: 
                    continue
                
                # Step 2.1: find Oldest Cell (Max Weight = Waiting Time)
                max_weight = -1
                for i in range(self.N):
                    if matched_in[i] == False and voq_status[i][j] > 0:
                        if voq_status[i][j] > max_weight:
                            max_weight = voq_status[i][j]
                
                if max_weight == -1:
                    continue
                
                # Step 2.2: find all inputs with the maximum weight
                candidates = []
                for i in range(self.N):
                    if matched_in[i] == False and voq_status[i][j] == max_weight:
                        candidates.append(i)
                
                # Step 2.3: tie-breaking
                selected_i = -1
                if len(candidates) == 1:
                    selected_i = candidates[0]
                else:
                    if self.tie_breaker == 'random':
                        rand_idx = random.randint(0, len(candidates) - 1)
                        selected_i = candidates[rand_idx]
                    else: 
                        for offset in range(self.N):
                            # index = (current pointer + offset) module N
                            idx = (self.out_ptr[j] + offset) % self.N
                            if idx in candidates:
                                selected_i = idx
                                break
                
                grants[j] = selected_i

            # Phase 3: Accept
            for i in range(self.N):
                if matched_in[i] == True: 
                    continue
                
                # Step 3.1: find the oldest cell among the grants received (Max Weight = Waiting Time)
                max_weight = -1
                for j in range(self.N):
                    if grants[j] == i:
                        if voq_status[i][j] > max_weight:
                            max_weight = voq_status[i][j]
                
                if max_weight == -1:
                    continue
                
                # Step 3.2: find all outputs with the maximum weight
                candidates = []
                for j in range(self.N):
                    if grants[j] == i and voq_status[i][j] == max_weight:
                        candidates.append(j)
                
                # Step 3.3: tie-breaking 
                selected_j = -1
                if len(candidates) == 1:
                    selected_j = candidates[0]
                else:
                    if self.tie_breaker == 'random':
                        rand_idx = random.randint(0, len(candidates) - 1)
                        selected_j = candidates[rand_idx]
                    else: 
                        for offset in range(self.N):
                            idx = (self.in_ptr[i] + offset) % self.N
                            if idx in candidates:
                                selected_j = idx
                                break
                
                match_result[i] = selected_j
                matched_in[i] = True
                matched_out[selected_j] = True
                
                # Pointer Updates 
                if self.tie_breaker == 'rr':
                    self.out_ptr[selected_j] = (i + 1) % self.N
                    if len(candidates) > 1:
                        self.in_ptr[i] = (selected_j + 1) % self.N

        return match_result
    
class iLPF(BaseScheduler):

    def __init__(self, N, num_iters=1, tie_breaker='rr'):
        super().__init__(N, num_iters)
        
        self.tie_breaker = tie_breaker

        """
        in_ptr: Round-robin pointer for each input port, used in the Accept phase.
        out_ptr: Round-robin pointer for each output port, used  in the Grant phase.
        """
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

        # Pre-calculation: calculate Port Occupancy for each input and output based on voq_status
        input_occupancy = []   # R_i (Row sum)
        output_occupancy = []  # C_j (Column sum)
        
        for idx in range(self.N):
            input_occupancy.append(0)
            output_occupancy.append(0)
            
        for i in range(self.N):
            for j in range(self.N):
                input_occupancy[i] += voq_status[i][j]
                output_occupancy[j] += voq_status[i][j]

        # Iterative Matching Process
        for iteration in range(self.num_iters):
            
            # --- Phase 1: Request ---

            # --- Phase 2: Grant ---
            grants = []
            for idx in range(self.N):
                grants.append(-1)

            for j in range(self.N):
                if matched_out[j] == True: 
                    continue
                
                # Step 2.1: find the maximum port weight (Max Weight = R_i + C_j)
                max_weight = -1
                for i in range(self.N):
                    # prerequisite: the queue must have data (voq_status[i][j] > 0)
                    if matched_in[i] == False and voq_status[i][j] > 0:
                        weight = input_occupancy[i] + output_occupancy[j]
                        if weight > max_weight:
                            max_weight = weight
                
                if max_weight == -1:
                    continue
                
                # Step 2.2: find all inputs with the maximum weight
                candidates = []
                for i in range(self.N):
                    if matched_in[i] == False and voq_status[i][j] > 0:
                        weight = input_occupancy[i] + output_occupancy[j]
                        if weight == max_weight:
                            candidates.append(i)
                
                # Step 2.3: tie-breaking
                selected_i = -1
                if len(candidates) == 1:
                    selected_i = candidates[0]
                else:
                    if self.tie_breaker == 'random':
                        rand_idx = random.randint(0, len(candidates) - 1)
                        selected_i = candidates[rand_idx]
                    else: 
                        for offset in range(self.N):
                            idx = (self.out_ptr[j] + offset) % self.N
                            if idx in candidates:
                                selected_i = idx
                                break
                
                grants[j] = selected_i

            # Phase 3: Accept 
            for i in range(self.N):
                if matched_in[i] == True: 
                    continue
                
                # Step 3.1: find the oldest cell among the grants received (Max Weight = Waiting Time)
                max_weight = -1
                for j in range(self.N):
                    if grants[j] == i:
                        weight = input_occupancy[i] + output_occupancy[j]
                        if weight > max_weight:
                            max_weight = weight
                
                if max_weight == -1:
                    continue
                
                # Step 3.2: find all outputs with the maximum weight
                candidates = []
                for j in range(self.N):
                    if grants[j] == i:
                        weight = input_occupancy[i] + output_occupancy[j]
                        if weight == max_weight:
                            candidates.append(j)
                
                # Step 3.3: tie-breaking
                selected_j = -1
                if len(candidates) == 1:
                    selected_j = candidates[0]
                else:
                    if self.tie_breaker == 'random':
                        rand_idx = random.randint(0, len(candidates) - 1)
                        selected_j = candidates[rand_idx]
                    else: 
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
