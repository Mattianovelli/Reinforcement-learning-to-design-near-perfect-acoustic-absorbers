import numpy as np
from air_features import air_features
air = air_features()

def crea_funzioni_G():    
    rho_0 = air.rho_0
    eta = air.eta
    Pr = air.Pr

    def G_rho(w):
        return np.sqrt(-1j * w * rho_0 / eta)

    def G_K(w):
        return np.sqrt(-1j * w * Pr * rho_0 / eta)

    return G_rho, G_K
