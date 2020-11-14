# First and second experiment: plots produced from stored data
# Copyright holders: Roland Maier, Barbara Verfuerth
# 2020

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio 

LOD_1 = sio.loadmat('data/_relErrEnergy')
LOD0_1 = sio.loadmat('data/_relErrEnergy0')
FEM_1 = sio.loadmat('data/_FEMrelErrEnergy')

LOD_2 = sio.loadmat('data/_Exp2relErrEnergy')
LOD0_2 = sio.loadmat('data/_Exp2relErrEnergy0')
FEM_2 = sio.loadmat('data/_Exp2FEMrelErrEnergy')

Hs = 0.5**np.arange(1,8)
its = np.arange(1,9)

errLOD_1 = np.min(LOD_1['relErrEnergy'],1)
errLOD0_1 = np.min(LOD0_1['relErrEnergy0'],1)
errFEM_1 = np.min(FEM_1['FEMrelErrEnergy'],1)

errLOD_1_3 = LOD_1['relErrEnergy'][2,:]
errLOD_1_4 = LOD_1['relErrEnergy'][3,:]
errLOD0_1_3 = LOD0_1['relErrEnergy0'][2,:]
errLOD0_1_4 = LOD0_1['relErrEnergy0'][3,:]
errFEM_1_3 = FEM_1['FEMrelErrEnergy'][2,:]
errFEM_1_4 = FEM_1['FEMrelErrEnergy'][3,:]

errLOD_2 = np.min(LOD_2['relErrEnergy'],1)
errLOD0_2 = np.min(LOD0_2['Exp2relErrEnergy0'],1)
errFEM_2 = np.min(FEM_2['Exp2FEMrelErrEnergy'],1)

plt.figure(1)
plt.title('Relative energy errors w.r.t H - Ex 1')
plt.plot(Hs,errLOD_1,'x-',color='blue', label='LOD_ad')
plt.plot(Hs,errLOD0_1,'x-',color='green', label='LOD_inf')
plt.plot(Hs,errFEM_1,'x-',color='red', label='FEM')
plt.plot([0.5,0.0078125],[0.5,0.0078125], color='black', linestyle='dashed', label='order 1')
plt.yscale('log')
plt.xscale('log')
plt.legend()

plt.figure(2)
plt.title('Relative energy errors w.r.t iterations - Ex 1')
plt.plot(its,errLOD_1_3[:8],'x-',color='blue', label='LOD_ad, Hlvl=3')
plt.plot(its,errLOD_1_4[:8],'x-',color='blue', label='LOD_ad, Hlvl=4', linestyle='dotted')
plt.plot(its,errLOD0_1_3[:8],'x-',color='green', label='LOD_inf, Hlvl=3')
plt.plot(its,errLOD0_1_4[:8],'x-',color='green', label='LOD_inf, Hlvl=4', linestyle='dotted')
plt.plot(its,errFEM_1_3[:8],'x-',color='red', label='FEM, Hlvl=3')
plt.plot(its,errFEM_1_4[:8],'x-',color='red', label='FEM, Hlvl=4', linestyle='dotted')
plt.yscale('log')
plt.legend()

plt.figure(3)
plt.title('Relative energy errors w.r.t H - Ex 2')
plt.plot(Hs,errLOD_2,'x-',color='blue', label='LOD_ad')
plt.plot(Hs,errLOD0_2,'x-',color='green', label='LOD_inf')
plt.plot(Hs,errFEM_2,'x-',color='red', label='FEM')
plt.plot([0.5,0.0078125],[0.75,0.01171875], color='black', linestyle='dashed', label='order 1')
plt.yscale('log')
plt.xscale('log')
plt.legend()

plt.show()
