import numpy as np
from crea_superficie import crea_superficie
from crea_rho_K import crea_rho_K

def crea_Z_k(oggetto):
    r = oggetto.r
    S = np.pi * r**2
    rho, K = crea_rho_K(oggetto)
    
    def Z(w):
        Z_val = np.sqrt(rho(w)*K(w))/S
        return Z_val

    def k(w):
        k_val = w*np.sqrt(rho(w)/K(w))
        return k_val

    return Z, k
