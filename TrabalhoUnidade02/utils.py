from math import e 

def arredondar_para_baixo(i):
    return int(i*100)/100

def tangenteHiperbolica(x):
    return ((1 - (e**(-2*x)))/(1 + (e**(-2*x))))
