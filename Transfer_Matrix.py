import numpy as np
from Codici_ausiliari.crea_Z_k import crea_Z_k

def TM(parameters):
    l = parameters.l
    
    Z, k = crea_Z_k(parameters)
    def Transfer_Matrix(w): 
        cos_ev = np.cos(k(w)*l)
        sin_ev = np.sin(k(w)*l)
        Z_ev = Z(w)
        return np.array([
            [cos_ev, 1j*Z_ev*sin_ev],[1j/Z_ev*sin_ev, cos_ev]], dtype=complex)
    return Transfer_Matrix



