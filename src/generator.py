import numpy as np

class JetSimulator:
    def __init__(self, num_events=10000):
        self.num_events = num_events

    def generate_signal(self, parent_mass=173.0, pz_boost=500.0):
        """Simulates Top Quark decay (Signal: Label 1)"""
        events = np.zeros((self.num_events, 2, 4)) 

        # 1. Decay in Rest Frame
        E_rest = parent_mass / 2.0
        phi = np.random.uniform(0, 2 * np.pi, self.num_events)
        costheta = np.random.uniform(-1, 1, self.num_events)
        theta = np.arccos(costheta)
        
        px = E_rest * np.sin(theta) * np.cos(phi)
        py = E_rest * np.sin(theta) * np.sin(phi)
        pz_rest = E_rest * np.cos(theta)
        
        # 2. Relativistic Kinematics
        E_parent = np.sqrt(pz_boost**2 + parent_mass**2)
        beta = pz_boost / E_parent 
        gamma = 1.0 / np.sqrt(1.0 - beta**2)
        
        # 3. Lorentz Boost
        events[:, 0, 0] = gamma * (E_rest + beta * pz_rest)       # E1
        events[:, 0, 1] = px                                      # px1
        events[:, 0, 2] = py                                      # py1
        events[:, 0, 3] = gamma * (pz_rest + beta * E_rest)       # pz1
        
        events[:, 1, 0] = gamma * (E_rest - beta * pz_rest)       # E2
        events[:, 1, 1] = -px                                     # px2
        events[:, 1, 2] = -py                                     # py2
        events[:, 1, 3] = gamma * (-pz_rest + beta * E_rest)      # pz2
        
        labels = np.ones(self.num_events)
        return events, labels

    def generate_background(self):
        """Simulates random QCD noise (Background: Label 0)"""
        events = np.zeros((self.num_events, 2, 4)) 

        # Randomize two independent particles
        for p_idx in [0, 1]:
            E = np.random.uniform(20.0, 150.0, self.num_events)
            theta = np.random.uniform(0, 0.5, self.num_events) 
            phi = np.random.uniform(0, 2 * np.pi, self.num_events)
            
            px = E * np.sin(theta) * np.cos(phi)
            py = E * np.sin(theta) * np.sin(phi)
            pz = E * np.cos(theta)
            
            # Apply generic forward boost to mimic jet momentum
            beta = 0.9
            gamma = 1.0 / np.sqrt(1.0 - beta**2)
            
            events[:, p_idx, 0] = gamma * (E + beta * pz)
            events[:, p_idx, 1] = px
            events[:, p_idx, 2] = py
            events[:, p_idx, 3] = gamma * (pz + beta * E)
            
        labels = np.zeros(self.num_events)
        return events, labels