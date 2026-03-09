import numpy as np

class LyapunovScheduler:
    """
    Lyapunov Drift-Plus-Penalty scheduler for Age of Channel State Information (AoCSI).
    Balances information freshness (AoCSI) with system throughput.
    """
    def __init__(self, n_users, V=10.0, max_pilots_per_slot=1):
        self.n_users = n_users
        # Trade-off parameter: URLLC prioritizes freshness (low V), eMBB prioritizes throughput (high V)
        if n_users == 2:
            self.V = np.array([0.5, 30.0])  # [URLLC, eMBB] weights
        else:
            self.V = np.array([V] * n_users)
        self.max_pilots_per_slot = max_pilots_per_slot
        
        # Virtual queues for AoCSI
        self.q_aocsi = np.zeros(n_users)
        
    def step(self, current_aocsi, estimated_rates):
        """
        Calculate Lyapunov drift-plus-penalty to select which users transmit pilots.
        
        current_aocsi: (n_users,) array of current age of CSI for each user
        Determines which users should transmit pilots.
        Objective: Minimize Drift-Plus-Penalty: dL - V * Rate
        """
        self.q_aocsi = np.array(current_aocsi)
        rates = np.array(estimated_rates)
        
        metrics = []
        for k in range(len(self.q_aocsi)):
            A = self.q_aocsi[k]
            R = rates[k]
            V_weight = self.V[k] if k < len(self.V) else self.V[-1]
            
            # Action 1: Send Pilot -> A(t+1) = 0, Rate = 0
            metric_pilot = -0.5 * (A**2)
            
            # Action 0: Send Data -> A(t+1) = A(t) + 1, Rate = R
            metric_data = (A + 0.5) - V_weight * R
            
            # We want to MINIMIZE the expected drift-plus-penalty
            diff = metric_pilot - metric_data
            metrics.append((k, diff))
            
        # Sort users by who gets the most negative benefit from sending a pilot
        metrics.sort(key=lambda x: x[1])
        
        pilot_users = []
        for i in range(min(self.max_pilots_per_slot, len(metrics))):
            if metrics[i][1] < 0:
                pilot_users.append(metrics[i][0])
                
        return pilot_users
