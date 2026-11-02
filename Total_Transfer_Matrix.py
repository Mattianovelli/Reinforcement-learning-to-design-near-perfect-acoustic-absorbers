import numpy as np
from Codici_ausiliari.Transfer_Matrix import TM
from functools import reduce

def Impedance(lista_parametri):
    TM_lista = [TM(p) for p in lista_parametri]
    def Z_r(w):
        # Total Transfer Matrix
        TTM = reduce(lambda A, B: A @ B, [tr_mat(w) for tr_mat in TM_lista])        

        # Componenti matrice
        T11 = TTM[0, 0]
        T21 = TTM[1, 0]

        # Normalized acoustic input impedance
        Z_r = T11 / T21

        return TTM, Z_r, Z_r.real, Z_r.imag
    return Z_r

