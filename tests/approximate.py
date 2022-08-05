import torch
import numpy as np

# No.3
def appro_tanhhh(x):
    p2 = x*x
    a = x*(10+p2)*(60+p2)
    b = (600+270*p2+11*p2*p2+p2*p2*p2/24)
    return a/b

# No.2
def appro_tanhh(x):
    return (-.67436811832e-5+(.2468149110712040+(.583691066395175e-1+.3357335044280075e-1*x)*x)*x)/(.2464845986383725+(.609347197060491e-1+(.1086202599228572+.2874707922475963e-1*x)*x)*x)

# No.1 this function neednt to segment [-4.97,4.97]
def appro_tanhhhh(x): 
    x2 = x * x
    a = x * (135135 + x2 * (17325 + x2 * (378 + x2)))
    b = 135135 + x2 * (62370 + x2 * (3150 + x2 * 28))
    return a / b

# Lowest computational complexity
def aprro_tanhhhhh(x):
    # return x*(10395+1260*x**2+21*x**4)/(10395+4725*x**2+210*x**4+x**6)
    return x*(945+105*x**2+x**4)/(945+420*x**2+15*x**4)

# this function neednt to segment, return:tanh(2/x)
def appro_tanhhalf(x): 
    # x = 2*x
    # [-4.5,4.5]
    # return 25.2075466924 * x * ( x**2 + 6.96321678**2 ) / (( x**2 + 3.17572786**2 ) * ( x**2 + 15.57611343**2 ))
    # [-10,10]
    return 94.9339451088 * x * ((( x**2 + 1.06315869e+03 ) * x**2 + 1.88748783e+05 ) * x**2 + 5.86237309e+06 ) / ((( ( x**2 + 4.03183926e+03 ) * x**2 + 1.64253046e+06 ) * x**2 + 1.28592857e+08 ) * x**2 + 1.11307745e+09 )

# def appro_sigmoid(x):
    # x[x>8] = 8
    # x[x<-8] = -8
    # return (appro_tanhhalf(x)+1)/2
    # # return (appro_tanhhhh(0.5*x)+1)/2

def appro_sigmoid(x):
    x[x>9.8] = 9.8
    x[x<-9.8] = -9.8
    return (appro_tanhhhh(0.5*x)+1)/2

def appro_tanh(x):
    x[x>4.9] = 4.9
    x[x<-4.9] = -4.9
    x = appro_tanhhhh(x)
    # x[x>0] = torch.FloatTensor(list(map(appro_tanhh,x[x>0])))
    # x[x<0] = -torch.FloatTensor(list(map(appro_tanhh,abs(x[x<0]))))
    return x
