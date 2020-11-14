# Second experiment
# Copyright holders: Roland Maier, Barbara Verfuerth
# 2020

# Set fineLevel=9, maxCoarseLevel=7, and maxIt=20 for the experiment in the paper.
# This might take some time.
# Note that the level of data oscillations is automatically set to fineLevel-2

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import scipy.io as sio 
import lodhelmholtz as lod

from gridlod import pglod, util, interp, coef, fem, func
from gridlod.world import World, Patch
from matplotlib import ticker

fineLevel = 7; 
maxCoarseLevel = 6;
maxIt = 8

def drawCoefficient(N, a):
    aCube = a.reshape(N, order='F')
    aCube = np.ascontiguousarray(aCube.T)
    plt.figure(6)
    
    cmap = plt.cm.get_cmap('binary')
            
    plt.imshow(aCube,
               origin='lower',
               interpolation='none',
               cmap=cmap,
               alpha=1)
    plt.title('Coefficient - Ex 2')
               
    #positions = (0, 127, 255) 
    #labels = ("0", "0.5", "1")
    #plt.xticks(positions, labels)
    #plt.yticks(positions, labels)

def helmholtz_nonlinear_adaptive(mapper,fineLvl,maxCoarseLvl,maxit):
    NFine = np.array([2 ** fineLvl, 2 ** fineLvl])
    NpFine = np.prod(NFine + 1)
    NList = 2**np.arange(1,maxCoarseLvl+1)
    ell = 2 # localization parameter

    k = 30. # wavenumber
    maxit_Fine = 250
    tol = 0.5 # coupled to maximal error indicator! (originally 0.02 and no coupling below!)

    xt = util.tCoordinates(NFine)
    xp = util.pCoordinates(NFine)

    # multiscale coefficients on the scale NFine-2
    np.random.seed(123)
    sizeK = np.size(xt[:, 0])
    nFine = NFine[0]

    # determine domain D_eps = supp(1-n) = supp(1-A) (all equal for this experiment)
    indicesIn = (xt[:, 0] > 0.25) & (xt[:, 0] < 0.75) & (xt[:, 1] > 0.25) & (xt[:, 1] < 0.75)  
    indicesInEps = (xt[:, 0] > 0.25) & (xt[:, 0] < 0.75) & (xt[:, 1] > 0.25) & (xt[:, 1] < 0.75) 

    # coefficients
    cA = .2 # lower bound on A
    CA = 1. # upper bound on A
    aEps = np.random.uniform(0, 1, sizeK // 16)
    aEpsPro = np.zeros(sizeK)
    for i in range((nFine) // 4):
        aEpsPro[4 * i * (nFine):4 * (i + 1) * (nFine)] = np.tile(np.repeat(aEps[i * (nFine) // 4:(i + 1) * (nFine) // 4], 4), 4)
    aFine = np.ones(xt.shape[0])
    aFine[indicesIn] = (CA - cA) * aEpsPro[indicesIn] + cA

    cn = 1. # lower bound on n
    Cn = 1. # upper bound on n
    nEps = np.random.uniform(0, 1, sizeK // 16)
    nEpsPro = np.zeros(sizeK)
    for i in range((nFine) // 4):
        nEpsPro[4 * i * (nFine):4 * (i + 1) * (nFine)] = np.tile(np.repeat(nEps[i * (nFine) // 4:(i + 1) * (nFine) // 4], 4), 4)
    
    k2Fine = k ** 2 * np.ones(xt.shape[0])
    k2Fine[indicesIn] = k ** 2 * ((Cn - cn) * nEpsPro[indicesIn] + cn)  
    kFine = k * np.ones(xt.shape[0]) 

    Ceps = .85 # upper bound on eps (lower bound is 0)
    lvl = 4
    epsEps = np.random.randint(2,size= (sizeK // lvl**2))
    epsEpsPro = np.zeros(sizeK)
    for i in range((nFine) // lvl):
        epsEpsPro[lvl * i * (nFine):lvl * (i + 1) * (nFine)] = np.tile(np.repeat(epsEps[i * (nFine) // lvl:(i + 1) * (nFine) // lvl], lvl), lvl)
    epsFine = np.zeros(xt.shape[0])
    epsFine[indicesInEps] = Ceps * epsEpsPro[indicesInEps]  ###  0 OR Ceps

    drawCoefficient(NFine,epsFine)

    xC = xp[:, 0]
    yC = xp[:, 1]
        
    fact = 100.
    mult = .8
    a = .5 
    b = .25 
    k2 = 30.

    # define right-hand side and boundary condition
    def funcF(x,y):
        res = mult*(-np.exp(-1.j*k2*(a*x-b))*(2*a**2*fact**2*np.sinh(fact*(a*x-b))**2/(np.cosh(fact*(a*x-b))+1)**3 - a**2*fact**2*np.cosh(fact*(a*x-b))/(np.cosh(fact*(a*x-b))+1)**2) + a**2*k2**2*np.exp(-1.j*k2*(a*x-b))/(np.cosh(fact*(a*x-b))+1) - 2.j*a**2*fact*k2*np.exp(-1.j*k2*(a*x-b))*np.sinh(fact*(a*x-b))/(np.cosh(fact*(a*x-b))+1)**2 - k**2*np.exp(-1.j*k2*(a*x-b))/(np.cosh(fact*(a*x-b))+1))
        return res
     
    f = funcF(xC,yC)

    g = np.zeros(NpFine, dtype='complex128')
    # bottom boundary
    g[0:(NFine[0]+1)] = mult*1.j*k*1./(np.cosh(fact*(a*xC[0:(NFine[0]+1)]-b))+1)*np.exp(-1.j*k2*(a*xC[0:(NFine[0]+1)]-b))
    # top boundary
    g[(NpFine-NFine[0]-1):] = mult*1.j*k*1./(np.cosh(fact*(a*xC[(NpFine-NFine[0]-1):NpFine]-b))+1)*np.exp(-1.j*k2*(a*xC[(NpFine-NFine[0]-1):NpFine]-b))
    # left boundary
    g[0:(NpFine-NFine[0]):(NFine[0]+1)] =  mult*1.j*k*np.ones_like(yC[0:(NpFine-NFine[0]):(NFine[0]+1)])/(np.cosh(fact*(a*0-b))+1)*np.exp(-1.j*k2*(a*0-b)) + mult*np.ones_like(yC[0:(NpFine-NFine[0]):(NFine[0]+1)])*(a*1.j*k2*np.exp(-1.j*k2*(a*0-b))/(np.cosh((a*0-b)*fact) +1) + a*fact*np.sinh((a*0-b)*fact)*np.exp(-1.j*k2*(a*0-b))/(np.cosh((a*0-b)*fact) +1)**2 )
    # right boundary
    g[NFine[0]:NpFine:(NFine[0]+1)] = mult*1.j*k*np.ones_like(yC[NFine[0]:NpFine:(NFine[0]+1)])/(np.cosh(fact*(a*1.-b))+1)*np.exp(-1.j*k2*(a*1.-b)) - mult*np.ones_like(yC[NFine[0]:NpFine:(NFine[0]+1)])*(a*1.j*k2*np.exp(-1.j*k2*(a*1.-b))/(np.cosh((a*1.-b)*fact) +1) + a*fact*np.sinh((a*1.-b)*fact)*np.exp(-1.j*k2*(a*1.-b))/(np.cosh((a*1.-b)*fact) +1)**2 )

    # reference solution
    uSol = np.zeros(NpFine, dtype='complex128')

    # boundary conditions
    boundaryConditions = np.array([[1, 1], [1, 1]])  # Robin boundary
    worldFine = World(NFine, np.array([1, 1]), boundaryConditions)

    # fine matrices
    BdFineFEM = fem.assemblePatchBoundaryMatrix(NFine, fem.localBoundaryMassMatrixGetter(NFine))
    MFineFEM = fem.assemblePatchMatrix(NFine, fem.localMassMatrix(NFine))
    KFineFEM = fem.assemblePatchMatrix(NFine, fem.localStiffnessMatrix(NFine))

    kBdFine = fem.assemblePatchBoundaryMatrix(NFine, fem.localBoundaryMassMatrixGetter(NFine), kFine)
    KFine = fem.assemblePatchMatrix(NFine, fem.localStiffnessMatrix(NFine), aFine)
    
    # incident beam
    uInc = mult/(np.cosh(fact*(a*xC-b))+1)*np.exp(-1.j*k2*(a*xC-b))

    print('***computing reference solution***')
    
    uOldFine = np.zeros(NpFine, dtype='complex128') 

    for it in np.arange(maxit_Fine):
        print('-- itFine = %d' % it)
        knonlinUpreFine = np.abs(uOldFine)
        knonlinUFine = func.evaluateCQ1(NFine, knonlinUpreFine, xt) 

        k2FineUfine = np.copy(k2Fine)
        k2FineUfine[indicesInEps] *= (1. + epsFine[indicesInEps] * knonlinUFine[indicesInEps] ** 2)  # full coefficient, including nonlinearity

        k2MFine = fem.assemblePatchMatrix(NFine, fem.localMassMatrix(NFine), k2FineUfine)  # weighted mass matrix, updated in every iteration

        nodesFine = np.arange(worldFine.NpFine)
        fixFine = util.boundarypIndexMap(NFine, boundaryConditions == 0)
        freeFine = np.setdiff1d(nodesFine, fixFine)

        # right-hand side (including boundary condition)
        fhQuad = MFineFEM * f + BdFineFEM * g

        # fine system
        lhsh = KFine[freeFine][:, freeFine] - k2MFine[freeFine][:, freeFine] + 1j * kBdFine[freeFine][:,freeFine]
        rhsh = fhQuad[freeFine]
        xFreeFine = sparse.linalg.spsolve(lhsh, rhsh)

        xFullFine = np.zeros(worldFine.NpFine, dtype='complex128')
        xFullFine[freeFine] = xFreeFine
        uOldFine = np.copy(xFullFine)

        # residual - used as stopping criterion
        knonlinU = np.abs(uOldFine)
        knonlinUFineIt = func.evaluateCQ1(NFine, knonlinU, xt)

        k2FineUfineIt = np.copy(k2Fine)
        k2FineUfineIt[indicesInEps] *= (1. + epsFine[indicesInEps] * knonlinUFineIt[indicesInEps] ** 2)  # update full coefficient, including nonlinearity

        k2MFineIt = fem.assemblePatchMatrix(NFine, fem.localMassMatrix(NFine), k2FineUfineIt)
        Ares = KFine - k2MFineIt + 1j * kBdFine
        residual = np.linalg.norm(Ares * xFullFine - fhQuad)/np.linalg.norm(Ares * xFullFine)
        print('---- residual = %.4e' % residual)
        
        if residual < 1e-12:
            break # stopping criterion

    uSol = xFullFine  # final fine reference solution

    print('***reference solution computed***\n')
    
    
######################################################################################    
    
    print('***computing multiscale approximations***')  
     
    relErrEnergy = np.zeros([len(NList),maxit])

    counter = 0
    for N in NList:
        counter += 1
        print('H = %.4e' % (1./N))
        NWorldCoarse = np.array([N, N])
        NCoarseElement = NFine // NWorldCoarse
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        NpCoarse = np.prod(NWorldCoarse + 1)

        uOldUps = np.zeros(NpFine, dtype='complex128')

        for it in np.arange(maxit):
            print('-- it = %d:' % it)
            knonlinUpre = np.abs(uOldUps)
            knonlinU = func.evaluateCQ1(NFine, knonlinUpre, xt)

            k2FineU = np.copy(k2Fine)
            k2FineU[indicesInEps] *= (1. + epsFine[indicesInEps] * knonlinU[indicesInEps] ** 2)

            print('---- starting computation of correctors')

            def computeLocalContribution(TInd):
                patch = Patch(world, ell, TInd)
                IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
                aPatch = lambda: coef.localizeCoefficient(patch, aFine)
                kPatch = lambda: coef.localizeCoefficient(patch, kFine)
                k2Patch = lambda: coef.localizeCoefficient(patch, k2FineU)

                correctorsList = lod.computeBasisCorrectors_helmholtz(patch, IPatch, aPatch, kPatch, k2Patch)
                csi = lod.computeBasisCoarseQuantities_helmholtz(patch, correctorsList, aPatch, kPatch, k2Patch)
                return patch, correctorsList, csi.Kmsij, csi.Mmsij, csi.Bdmsij, csi.muTPrime

            def computeIndicators(TInd):
                k2FineUPatch = lambda: coef.localizeCoefficient(patchT[TInd], k2FineU)
                k2FineUOldPatch = lambda: coef.localizeCoefficient(patchT[TInd], k2FineUOld)

                E_vh = lod.computeErrorIndicatorCoarse_helmholtz(patchT[TInd],muTPrime[TInd],k2FineUOldPatch,k2FineUPatch)
                return E_vh

            def UpdateCorrectors(TInd):
                patch = Patch(world, ell, TInd)
                IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
                aPatch = lambda: coef.localizeCoefficient(patch, aFine)
                kPatch = lambda: coef.localizeCoefficient(patch, kFine)
                k2Patch = lambda: coef.localizeCoefficient(patch, k2FineU)

                correctorsList = lod.computeBasisCorrectors_helmholtz(patch, IPatch, aPatch, kPatch, k2Patch)
                csi = lod.computeBasisCoarseQuantities_helmholtz(patch, correctorsList, aPatch, kPatch, k2Patch)
                return patch, correctorsList, csi.Kmsij, csi.Mmsij, csi.Bdmsij, csi.muTPrime

            def UpdateElements(tol, E, Kmsij_old, Mmsij_old, Bdmsij_old, correctors_old, mu_old):
                print('---- apply tolerance')
                Elements_to_be_updated = []
                for (i, eps) in E.items():
                    if eps > tol :
                        Elements_to_be_updated.append(i)
                if len(E) > 0:
                    print('---- total percentage of element correctors to be updated: %.4f' % (100 * np.size(Elements_to_be_updated) / len(mu_old)), flush=True)

                print('---- update local contributions')
                KmsijT_list = list(np.copy(Kmsij_old))
                MmsijT_list = list(np.copy(Mmsij_old))
                BdmsijT_list = list(np.copy(Bdmsij_old))
                muT_list = np.copy(mu_old)
                for T in np.setdiff1d(range(world.NtCoarse), Elements_to_be_updated):
                    patch = Patch(world, ell, T)
                    aPatch = lambda: coef.localizeCoefficient(patch, aFine)
                    kPatch = lambda: coef.localizeCoefficient(patch, kFine)
                    k2Patch = lambda: coef.localizeCoefficient(patch, k2FineU)
                    csi = lod.computeBasisCoarseQuantities_helmholtz(patch, correctors_old[T], aPatch, kPatch, k2Patch)

                    KmsijT_list[T] = csi.Kmsij
                    MmsijT_list[T] = csi.Mmsij
                    BdmsijT_list[T] = csi.Bdmsij
                    muT_list[T] = csi.muTPrime

                if np.size(Elements_to_be_updated) != 0:
                    #print('---- update correctors')
                    patchT_irrelevant, correctorsListTNew, KmsijTNew, MmsijTNew, BdmsijTNew, muTPrimeNew = zip(*mapper(UpdateCorrectors,Elements_to_be_updated))

                    #print('---- update correctorsList')
                    correctorsListT_list = list(np.copy(correctors_old))
                    i = 0
                    for T in Elements_to_be_updated:
                        KmsijT_list[T] = KmsijTNew[i]
                        correctorsListT_list[T] = correctorsListTNew[i]
                        MmsijT_list[T] = MmsijTNew[i]
                        BdmsijT_list[T] = BdmsijTNew[i]
                        muT_list[T] = muTPrimeNew[i]
                        i += 1

                    KmsijT = tuple(KmsijT_list)
                    correctorsListT = tuple(correctorsListT_list)
                    MmsijT = tuple(MmsijT_list)
                    BdmsijT = tuple(BdmsijT_list)
                    muTPrime = tuple(muT_list)
                    return correctorsListT, KmsijT, MmsijT, BdmsijT, muTPrime
                else:
                    KmsijT = tuple(KmsijT_list)
                    MmsijT = tuple(MmsijT_list)
                    BdmsijT = tuple(BdmsijT_list)
                    muTPrime = tuple(muT_list)
                    return correctors_old, KmsijT, MmsijT, BdmsijT, muTPrime

            if it == 0:
                patchT, correctorsListT, KmsijT, MmsijT, BdmsijT, muTPrime = zip(
                    *mapper(computeLocalContribution, range(world.NtCoarse)))
            else:
                E_vh = list(mapper(computeIndicators, range(world.NtCoarse)))
                print('---- maximal value error estimator for basis correctors {}'.format(np.max(E_vh)))
                E = {i: E_vh[i] for i in range(np.size(E_vh)) if E_vh[i] > 0 }

                # loop over elements with possible recomputation of correctors
                correctorsListT, KmsijT, MmsijT, BdmsijT, muTPrime = UpdateElements(tol*np.max(E_vh), E, KmsijT, MmsijT, BdmsijT, correctorsListT, muTPrime) # tol scaled by maximal error indicator

            print('---- finished computation of correctors')

            KLOD = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT) # ms stiffness matrix
            k2MLOD = pglod.assembleMsStiffnessMatrix(world, patchT, MmsijT)  # ms mass matrix
            kBdLOD = pglod.assembleMsStiffnessMatrix(world, patchT, BdmsijT)  # ms boundary matrix
            MFEM = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)
            BdFEM = fem.assemblePatchBoundaryMatrix(NWorldCoarse, fem.localBoundaryMassMatrixGetter(NWorldCoarse))
            print('---- coarse matrices assembled')

            nodes = np.arange(world.NpCoarse)
            fix = util.boundarypIndexMap(NWorldCoarse, boundaryConditions == 0)
            free = np.setdiff1d(nodes, fix)
            assert (nodes.all() == free.all())

            # compute global interpolation matrix
            patchGlobal = Patch(world, NFine[0] + 2, 0)
            IH = interp.L2ProjectionPatchMatrix(patchGlobal, boundaryConditions)
            assert (IH.shape[0] == NpCoarse)

            basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)

            fHQuad = basis.T * MFineFEM * f + basis.T*BdFineFEM*g

            print('---- solving coarse system')

            # coarse system
            lhsH = KLOD[free][:, free] - k2MLOD[free][:, free] + 1j * kBdLOD[free][:,free]
            rhsH = fHQuad[free]
            xFree = sparse.linalg.spsolve(lhsH, rhsH)

            basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
            modifiedBasis = basis - basisCorrectors

            xFull = np.zeros(world.NpCoarse, dtype='complex128')
            xFull[free] = xFree
            uLodCoarse = basis * xFull
            uLodFine = modifiedBasis * xFull
            uOldUps = np.copy(uLodFine)
            k2FineUOld = np.copy(k2FineU)

            # visualization
            if it == maxit - 1 and N == 2**1:
                grid = uLodFine.reshape(NFine + 1, order='C')

                plt.figure(2)
                plt.title('LOD_ad, Hlvl=4 - Ex 2')
                plt.imshow(grid.real, extent=(xC.min(), xC.max(), yC.min(), yC.max()), cmap=plt.cm.hot,origin='lower', vmin = -.6, vmax = .6)
                plt.colorbar()

                grid2 = uSol.reshape(NFine + 1, order='C')

                plt.figure(1)
                plt.title('reference solution - Ex 2')
                plt.imshow(grid2.real, extent=(xC.min(), xC.max(), yC.min(), yC.max()), cmap=plt.cm.hot,origin='lower', vmin = -.6, vmax = .6)
                plt.colorbar()
                
                grid3 = uInc.reshape(NFine + 1, order='C')

                plt.figure(6)
                plt.title('incident beam - Ex 2')
                plt.imshow(grid3.real, extent=(xC.min(), xC.max(), yC.min(), yC.max()), cmap=plt.cm.hot,origin='lower', vmin = -.6, vmax = .6)
                plt.colorbar()

            Err = np.sqrt(np.dot((uSol - uLodFine).conj(), KFineFEM * (uSol - uLodFine)) + k**2*np.dot((uSol - uLodFine).conj(), MFineFEM * (uSol - uLodFine)))
            ErrEnergy = Err / np.sqrt(np.dot((uSol).conj(), KFineFEM * (uSol)) + k**2*np.dot((uSol).conj(), MFineFEM * (uSol)))
            print('---- ',np.abs(ErrEnergy), '\n***********************************************')
            
            # save errors in arrays
            relErrEnergy[counter-1,it] = ErrEnergy     
            
        print('\n')

    
######################################################################################      
    
    print('***computing multiscale approximations without updates of correctors***')
    
    relErrEnergyNoUpdate = np.zeros([len(NList),maxit])
    
    counter = 0
    for N in NList:
        counter += 1
        print('H = %.4e' % (1./N))
        NWorldCoarse = np.array([N, N])
        NCoarseElement = NFine // NWorldCoarse
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        NpCoarse = np.prod(NWorldCoarse + 1)

        uOldUps = np.zeros(NpFine, dtype='complex128')

        for it in np.arange(maxit):
            print('-- it = %d:' % it)
            knonlinUpre = np.abs(uOldUps)
            knonlinU = func.evaluateCQ1(NFine, knonlinUpre, xt)

            k2FineU = np.copy(k2Fine)
            k2FineU[indicesInEps] *= (1. + epsFine[indicesInEps] * knonlinU[indicesInEps] ** 2)

            print('---- starting computation of correctors')

            def computeLocalContribution(TInd):
                patch = Patch(world, ell, TInd)
                IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
                aPatch = lambda: coef.localizeCoefficient(patch, aFine)
                kPatch = lambda: coef.localizeCoefficient(patch, kFine)
                k2Patch = lambda: coef.localizeCoefficient(patch, k2FineU)

                correctorsList = lod.computeBasisCorrectors_helmholtz(patch, IPatch, aPatch, kPatch, k2Patch)  # adapted for Helmholtz setting
                csi = lod.computeBasisCoarseQuantities_helmholtz(patch, correctorsList, aPatch, kPatch, k2Patch)  # adapted for Helmholtz setting
                return patch, correctorsList, csi.Kmsij, csi.Mmsij, csi.Bdmsij, csi.muTPrime

            def computeIndicators(TInd):
                k2FineUPatch = lambda: coef.localizeCoefficient(patchT[TInd], k2FineU)
                k2FineUOldPatch = lambda: coef.localizeCoefficient(patchT[TInd], k2FineUOld)

                E_vh = lod.computeErrorIndicatorCoarse_helmholtz(patchT[TInd],muTPrime[TInd],k2FineUOldPatch,k2FineUPatch)
                return E_vh

            def UpdateCorrectors(TInd):
                patch = Patch(world, ell, TInd)
                IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
                aPatch = lambda: coef.localizeCoefficient(patch, aFine)
                kPatch = lambda: coef.localizeCoefficient(patch, kFine)
                k2Patch = lambda: coef.localizeCoefficient(patch, k2FineU)

                correctorsList = lod.computeBasisCorrectors_helmholtz(patch, IPatch, aPatch, kPatch, k2Patch)
                csi = lod.computeBasisCoarseQuantities_helmholtz(patch, correctorsList, aPatch, kPatch, k2Patch)  # adapted for Helmholtz setting
                return patch, correctorsList, csi.Kmsij, csi.Mmsij, csi.Bdmsij, csi.muTPrime

            def UpdateElements(tol, E, Kmsij_old, Mmsij_old, Bdmsij_old, correctors_old, mu_old):
                print('---- apply tolerance')
                Elements_to_be_updated = []
                for (i, eps) in E.items():
                    if eps > tol : 
                        Elements_to_be_updated.append(i)
                if len(E) > 0:
                    print('---- total percentage of element correctors to be updated: %.4f' % (100 * np.size(Elements_to_be_updated) / len(mu_old)), flush=True)

                print('---- update local contributions')
                KmsijT_list = list(np.copy(Kmsij_old))
                MmsijT_list = list(np.copy(Mmsij_old))
                BdmsijT_list = list(np.copy(Bdmsij_old))
                muT_list = np.copy(mu_old)
                for T in np.setdiff1d(range(world.NtCoarse), Elements_to_be_updated):
                    patch = Patch(world, ell, T)
                    aPatch = lambda: coef.localizeCoefficient(patch, aFine)
                    kPatch = lambda: coef.localizeCoefficient(patch, kFine)
                    k2Patch = lambda: coef.localizeCoefficient(patch, k2FineU)
                    csi = lod.computeBasisCoarseQuantities_helmholtz(patch, correctors_old[T], aPatch, kPatch, k2Patch)

                    KmsijT_list[T] = csi.Kmsij
                    MmsijT_list[T] = csi.Mmsij
                    BdmsijT_list[T] = csi.Bdmsij
                    muT_list[T] = csi.muTPrime

                if np.size(Elements_to_be_updated) != 0:
                    #print('---- update correctors')
                    patchT_irrelevant, correctorsListTNew, KmsijTNew, MmsijTNew, BdmsijTNew, muTPrimeNew = zip(*mapper(UpdateCorrectors,Elements_to_be_updated))

                    #print('---- update correctorsList')
                    correctorsListT_list = list(np.copy(correctors_old))
                    i = 0
                    for T in Elements_to_be_updated:
                        KmsijT_list[T] = KmsijTNew[i]
                        correctorsListT_list[T] = correctorsListTNew[i]
                        MmsijT_list[T] = MmsijTNew[i]
                        BdmsijT_list[T] = BdmsijTNew[i]
                        muT_list[T] = muTPrimeNew[i]
                        i += 1

                    KmsijT = tuple(KmsijT_list)
                    correctorsListT = tuple(correctorsListT_list)
                    MmsijT = tuple(MmsijT_list)
                    BdmsijT = tuple(BdmsijT_list)
                    muTPrime = tuple(muT_list)
                    return correctorsListT, KmsijT, MmsijT, BdmsijT, muTPrime
                else:
                    KmsijT = tuple(KmsijT_list)
                    MmsijT = tuple(MmsijT_list)
                    BdmsijT = tuple(BdmsijT_list)
                    muTPrime = tuple(muT_list)
                    return correctors_old, KmsijT, MmsijT, BdmsijT, muTPrime

            if it == 0:
                patchT, correctorsListT, KmsijT, MmsijT, BdmsijT, muTPrime = zip(
                    *mapper(computeLocalContribution, range(world.NtCoarse)))
            else:
                E_vh = list(mapper(computeIndicators, range(world.NtCoarse)))
                print('---- maximal value error estimator for basis correctors {}'.format(np.max(E_vh)))
                E = {i: E_vh[i] for i in range(np.size(E_vh)) if E_vh[i] > 0 }

                # loop over elements with possible recomputation of correctors
                correctorsListT, KmsijT, MmsijT, BdmsijT, muTPrime = UpdateElements(2.*np.max(E_vh), E, KmsijT, MmsijT, BdmsijT, correctorsListT, muTPrime) # no updates

            print('---- finished computation of correctors')

            KLOD = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT) # ms stiffness matrix
            k2MLOD = pglod.assembleMsStiffnessMatrix(world, patchT, MmsijT)  # ms mass matrix
            kBdLOD = pglod.assembleMsStiffnessMatrix(world, patchT, BdmsijT)  # ms boundary matrix
            MFEM = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)
            BdFEM = fem.assemblePatchBoundaryMatrix(NWorldCoarse, fem.localBoundaryMassMatrixGetter(NWorldCoarse))
            print('---- coarse matrices assembled')

            nodes = np.arange(world.NpCoarse)
            fix = util.boundarypIndexMap(NWorldCoarse, boundaryConditions == 0)
            free = np.setdiff1d(nodes, fix)
            assert (nodes.all() == free.all())

            # compute global interpolation matrix
            patchGlobal = Patch(world, NFine[0] + 2, 0)
            IH = interp.L2ProjectionPatchMatrix(patchGlobal, boundaryConditions)
            assert (IH.shape[0] == NpCoarse)

            basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)

            fHQuad = basis.T * MFineFEM * f + basis.T*BdFineFEM*g

            print('---- solving coarse system')

            # coarse system
            lhsH = KLOD[free][:, free] - k2MLOD[free][:, free] + 1j * kBdLOD[free][:,free]
            rhsH = fHQuad[free]
            xFree = sparse.linalg.spsolve(lhsH, rhsH)

            basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
            modifiedBasis = basis - basisCorrectors

            xFull = np.zeros(world.NpCoarse, dtype='complex128')
            xFull[free] = xFree
            uLodCoarse = basis * xFull
            uLodFine = modifiedBasis * xFull
            uOldUps = np.copy(uLodFine)
            k2FineUOld = np.copy(k2FineU)
            
            resCoarse = np.linalg.norm((KLOD-k2MLOD+1j*kBdLOD) * xFull - fHQuad)

            # visualization
            if it == maxit - 1 and N == 2**4:
                grid = uLodFine.reshape(NFine + 1, order='C')

                plt.figure(3)
                plt.title('LOD_inf, Hlvl=4 - Ex 2')
                plt.imshow(grid.real, extent=(xC.min(), xC.max(), yC.min(), yC.max()), cmap=plt.cm.hot,origin='lower', vmin = -.6, vmax = .6)
                plt.colorbar()

            Err = np.sqrt(np.dot((uSol - uLodFine).conj(), KFineFEM * (uSol - uLodFine)) + k**2*np.dot((uSol - uLodFine).conj(), MFineFEM * (uSol - uLodFine)))
            ErrEnergy = Err / np.sqrt(np.dot((uSol).conj(), KFineFEM * (uSol)) + k**2*np.dot((uSol).conj(), MFineFEM * (uSol)))
            print('---- ',np.abs(ErrEnergy), '\n***********************************************')
            
            # save errors in arrays
            relErrEnergyNoUpdate[counter-1,it] = ErrEnergy
            
        print('\n')

 
######################################################################################     
        
    print('***computing FEM approximations***')
    
    FEMrelErrEnergy = np.zeros([len(NList),maxit])
    
    counter = 0
    for N in NList:
        counter += 1
        print('H = %.4e' % (1./N))
        NWorldCoarse = np.array([N, N])
        NCoarseElement = NFine // NWorldCoarse
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        NpCoarse = np.prod(NWorldCoarse + 1)
        
        xT = util.tCoordinates(NWorldCoarse)
        xP = util.pCoordinates(NWorldCoarse)

        uOld = np.zeros(NpCoarse, dtype='complex128')
        
        # compute coarse coefficients by averaging        
        NtC = np.prod(NWorldCoarse)
        aCoarse = np.zeros(NtC)
        kCoarse = k * np.ones(xT.shape[0])
        k2Coarse = np.zeros(NtC)
        epsCoarse = np.zeros(NtC)
        for Q in range(NtC):
            patch = Patch(world,0,Q)            
            aPatch = coef.localizeCoefficient(patch, aFine)
            epsPatch = coef.localizeCoefficient(patch, epsFine)
            k2Patch = coef.localizeCoefficient(patch, k2Fine)
            
            aCoarse[Q] = np.sum(aPatch)/(len(aPatch))
            k2Coarse[Q] = np.sum(k2Patch)/(len(k2Patch))
            epsCoarse[Q] = np.sum(epsPatch)/(len(epsPatch))        

        # coarse matrices
        KFEM = fem.assemblePatchMatrix(NWorldCoarse, fem.localStiffnessMatrix(NWorldCoarse), aCoarse)
        kBdFEM = fem.assemblePatchBoundaryMatrix(NWorldCoarse, fem.localBoundaryMassMatrixGetter(NWorldCoarse), kCoarse)
        MFEM = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)
        BdFEM = fem.assemblePatchBoundaryMatrix(NWorldCoarse, fem.localBoundaryMassMatrixGetter(NWorldCoarse))

        for it in np.arange(maxit):
            print('-- it = %d:' % it)
            knonlinUpre = np.abs(uOld)
            knonlinU = func.evaluateCQ1(NWorldCoarse, knonlinUpre, xT)

            k2CoarseU = np.copy(k2Coarse)
            k2CoarseU *= (1. + epsCoarse * knonlinU ** 2)
            
            # update weighted mass matrix
            k2MFEM = fem.assemblePatchMatrix(NWorldCoarse, fem.localMassMatrix(NWorldCoarse),k2CoarseU)
            
            nodes = np.arange(world.NpCoarse)
            fix = util.boundarypIndexMap(NWorldCoarse, boundaryConditions == 0)
            free = np.setdiff1d(nodes, fix)
            assert (nodes.all() == free.all())

            basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)

            fHQuad = basis.T * MFineFEM * f + basis.T*BdFineFEM*g

            print('---- solving coarse system')

            # coarse system
            lhsH = KFEM[free][:, free] - k2MFEM[free][:, free] + 1j * kBdFEM[free][:,free]
            rhsH = fHQuad[free]
            xFree = sparse.linalg.spsolve(lhsH, rhsH)

            xFull = np.zeros(world.NpCoarse, dtype='complex128')
            xFull[free] = xFree
            uCoarseInt = basis * xFull
            uOld = np.copy(xFull)

            # visualization
            if it == maxit - 1 and N == 2**4:
                grid = uCoarseInt.reshape(NFine + 1, order='C')

                plt.figure(4)
                plt.title('FEM, Hlvl=4 - Ex 2')
                plt.imshow(grid.real, extent=(xC.min(), xC.max(), yC.min(), yC.max()), cmap=plt.cm.hot, origin='lower', vmin = -.6, vmax = .6)
                plt.colorbar()

            Err = np.sqrt(np.dot((uSol - uCoarseInt).conj(), KFineFEM * (uSol - uCoarseInt)) + k**2*np.dot((uSol - uCoarseInt).conj(), MFineFEM * (uSol - uCoarseInt)))
            ErrEnergy = Err / np.sqrt(np.dot((uSol).conj(), KFineFEM * (uSol)) + k**2*np.dot((uSol).conj(), MFineFEM * (uSol)))
            print('---- ',np.abs(ErrEnergy), '\n***********************************************')
            
            # save errors in arrays
            FEMrelErrEnergy[counter-1,it] = ErrEnergy 
            
        print('\n')
    
    # error plots
    errLOD_2 = np.min(relErrEnergy,1)
    errLOD0_2 = np.min(relErrEnergyNoUpdate,1)
    errFEM_2 = np.min(FEMrelErrEnergy,1)
    
    Hs = 0.5**np.arange(1,maxCoarseLvl+1)
    
    plt.figure(5)
    plt.title('Relative energy errors w.r.t H - Ex 2')
    plt.plot(Hs,errLOD_2,'x-',color='blue', label='LOD_ad')
    plt.plot(Hs,errLOD0_2,'x-',color='green', label='LOD_inf')
    plt.plot(Hs,errFEM_2,'x-',color='red', label='FEM')
    plt.plot([0.5,0.0078125],[0.75,0.01171875], color='black', linestyle='dashed', label='order 1')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()

    plt.show()

# run the code
helmholtz_nonlinear_adaptive(map, fineLevel, maxCoarseLevel, maxIt)
