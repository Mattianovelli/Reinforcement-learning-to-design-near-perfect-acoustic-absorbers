## AIR FEATURES
# 

from dataclasses import dataclass
import numpy as np

@dataclass
class AirData:
    atemp: float
    P0: float       # ambient pressure [Pa]
    gamma: float    # ratio of specific heats
    rho_0: float    # density [kg/m^3]
    K0: float       # bulk modulus of air
    eta: float      # dynamic viscosity
    Pr: float       # Prandtl number
    c0: float       # speed of sound [m/s]
    Z0: float       # Impedance

def air_features(atemp: float = 1.0) -> AirData:
    
    P0 = 101325           # Pa
    gamma = 1.4
    rho_0 = 1.213          # kg/m^3
    K0 = gamma * P0        # bulk modulus
    eta = atemp * 1.839e-5 # dynamic viscosity
    Pr = atemp * 0.71      # Prandtl number
    c0 = np.sqrt(gamma * P0 / rho_0)  # speed of sound
    Z0 = rho_0 * c0        # Impedance

    return AirData(
        atemp = atemp,
        P0 = P0,
        gamma = gamma,
        rho_0 = rho_0,
        K0 = K0,
        eta = eta,
        Pr = Pr,
        c0 = c0,
        Z0 = Z0
    )
