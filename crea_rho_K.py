import numpy as np
from scipy.special import jv
from Codici_ausiliari.G import crea_funzioni_G
from Codici_ausiliari.air_features import air_features
air = air_features()

def crea_rho_K(oggetto):
    # Parametri dell’aria
    rho_0 = air.rho_0
    K0 = air.K0
    gamma = air.gamma

    # Funzioni G_rho e G_K
    G_rho, G_K = crea_funzioni_G(geom)

    # Calcolo di rho e K
    r = oggetto.r
    def rho(w):
        rG = r * G_rho(w)
        with np.errstate(all='ignore'):
            val = rho_0 * (1 - 2 / rG * jv(1, rG) / jv(0, rG))**(-1)
        if not np.any(np.isfinite(val)):
            return rho_0
        return val

    def K(w):
        rG = r * G_K(w)
        with np.errstate(all='ignore'):
            val = K0 * (1 + 2 * (gamma - 1) / rG * jv(1, rG) / jv(0, rG))**(-1)
        if not np.any(np.isfinite(val)):
            return K0
        return val
    
    return rho, K
