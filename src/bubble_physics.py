import numpy as np
from scipy.integrate import odeint
from typing import Tuple, List

class BubblePhysics:
    def __init__(self, R0=1e-6, P_inf=101325, rho=998, sigma=0.072, mu=0.001):
        self.R0 = R0  # Initial radius
        self.P_inf = P_inf  # Ambient pressure
        self.rho = rho  # Liquid density
        self.sigma = sigma  # Surface tension
        self.mu = mu  # Dynamic viscosity
        
    def rayleigh_plesset(self, state: List[float], t: float, P_v: float) -> Tuple[float, float]:
        R, Rdot = state
        
        # Rayleigh-Plesset equation terms
        P_g = self.P_inf * (self.R0/R)**3
        P_L = P_g + P_v - self.P_inf - 2*self.sigma/R
        
        Rddot = (P_L/(self.rho) - 3*Rdot**2/(2*R) - 4*self.mu*Rdot/(self.rho*R**2))
        return [Rdot, Rddot]

    def simulate(self, t_span: np.ndarray, P_v: float) -> Tuple[np.ndarray, np.ndarray]:
        initial_state = [self.R0, 0]
        solution = odeint(self.rayleigh_plesset, initial_state, t_span, args=(P_v,))
        return solution[:, 0], solution[:, 1]  # R(t), Rdot(t)
