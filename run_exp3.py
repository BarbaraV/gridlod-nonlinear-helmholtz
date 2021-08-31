# Third experiment
# Copyright holders: Roland Maier, Barbara Verfuerth
# 2021
#
# Choose fineLevel=9, coarseLevel=3, and maxIt=8 for the experiment in the paper.
# This will take time since the algorithm below does not exploit the potential
# parallelization and computes the different approximations one after the other.
# To obtain a first qualitative impression, it is recommended to use the parameters
# fineLevel=7, coarseLevel=3, and maxIt=12.
#
# Note that the level of data oscillations is automatically set to fineLevel-2

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from gridlod import pglod, util, lod, interp, coef, fem, func
from gridlod.world import World, Patch

fineLevel = 9 
coarseLevel = 3
maxIt = 8 

def coeffi(x,y,fineExp):
    epslevel = fineExp-2 
    epslevel2 = fineExp-4
    alpha = 0.
    beta = 1.
    res = np.ones(x.shape)
    np.random.seed(123)
    ran = np.random.uniform(0,1,x.shape[0]//(2**(2*epslevel2)))
    eps = 1e-15
    indices = (x<1+eps)*(x>-eps)*(y<1+eps)*(y>-eps)*(np.floor(2**(epslevel)*x)+2**epslevel*(np.floor(2**(epslevel)*y)))
    for i in range(indices.shape[0]):
        res[i] = alpha + (beta-alpha)*(1+np.sin(ran[int(indices[i])//64]*1.29408694*(indices[i])))
    return res


def drawCoefficient(N, a):
    aCube = a.reshape(N, order='F')
    aCube = np.ascontiguousarray(aCube.T)
    plt.figure(2)
    
    cmap = plt.cm.get_cmap('binary')
            
    plt.imshow(aCube,
               origin='lower',
               interpolation='none',
               cmap=cmap,
               alpha=1)
    plt.title('Coefficient - Ex 3')
               
    #positions = (0, 127, 255)
    #labels = ("0", "0.5", "1")
    #plt.xticks(positions, labels)
    #plt.yticks(positions, labels)
    
def helmholtz_nonlinear_adaptive(mapper,fineLvl,coarseLvl,maxit):
    fineExp = fineLvl
    NFine = np.array([2 ** fineLvl, 2 ** fineLvl])
    NpFine = np.prod(NFine + 1)
    N = 2**coarseLvl
    tolList = [2.0,1.0,0.5,0.25,0.125,0.0625,0.]
    ell = 2 # localization parameter

    k = 15. # wavenumber
    maxit_Fine = 200

    xt = util.tCoordinates(NFine)
    xp = util.pCoordinates(NFine)

    # multiscale coefficients on the scale NFine-2
    np.random.seed(444)
    sizeK = np.size(xt[:, 0])
    nFine = NFine[0]

    # determine domain D_eps = supp(1-n) = supp(1-A) (all equal for the moment)
    indicesIn = (xt[:, 0] > 0.15) & (xt[:, 0] < 0.85) & (xt[:, 1] > 0.15) & (xt[:, 1] < 0.85)  
    indicesInEps = (xt[:, 0] > 0.15) & (xt[:, 0] < 0.85) & (xt[:, 1] > 0.15) & (xt[:, 1] < 0.85) 

    # coefficients
    aFine = np.ones(xt.shape[0])

    cn = .05 # lower bound on n
    Cn = 1. # upper bound on n
    nEpsPro = coeffi(xt[:,0],xt[:,1],fineLvl)
    
    k2Fine = k ** 2 * np.ones(xt.shape[0])
    k2Fine[indicesIn] = k ** 2 * ((Cn - cn) * nEpsPro[indicesIn] + cn)  
    kFine = k * np.ones(xt.shape[0])  

    Ceps = 0.3 # upper bound on eps (lower bound is 0)
    epsEpsPro = np.ones(sizeK) 
    epsFine = np.zeros(xt.shape[0])
    epsFine[indicesInEps] = Ceps * epsEpsPro[indicesInEps]  # 0 OR Ceps
    
    plotC = np.ones(sizeK)
    plotC[indicesIn] = nEpsPro[indicesIn]
    drawCoefficient(NFine,plotC)

    xC = xp[:, 0]
    yC = xp[:, 1]

    # define right-hand side and boundary condition
    def funcF(x, y):
        res = 100*np.ones(x.shape, dtype='complex128') 
        return res
    f = funcF(xC, yC)
    
    # reference solution
    uSol = np.zeros(NpFine, dtype='complex128')

    # boundary conditions
    boundaryConditions = np.array([[1, 1], [1, 1]]) 
    worldFine = World(NFine, np.array([1, 1]), boundaryConditions)

    # fine matrices
    BdFineFEM = fem.assemblePatchBoundaryMatrix(NFine, fem.localBoundaryMassMatrixGetter(NFine))
    MFineFEM = fem.assemblePatchMatrix(NFine, fem.localMassMatrix(NFine))
    KFineFEM = fem.assemblePatchMatrix(NFine, fem.localStiffnessMatrix(NFine))  # , aFine)

    kBdFine = fem.assemblePatchBoundaryMatrix(NFine, fem.localBoundaryMassMatrixGetter(NFine), kFine)
    KFine = fem.assemblePatchMatrix(NFine, fem.localStiffnessMatrix(NFine), aFine)

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

        # right-hand side
        fhQuad = MFineFEM * f

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

    counter = 0  # for figures
    
    print('***computing multiscale approximations***')
    
    relErrEnergy = np.zeros([len(tolList),maxit])

    for tol in tolList:
        counter += 1
        print('H = %.4e, tol = %.4e' % (1./N,tol))
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
                    if eps > tol*k**2 : 
                        Elements_to_be_updated.append(i)
                if len(E) > 0:
                    print('---- percentage of non-zero element correctors to be updated: %.4f' % (100 * np.size(Elements_to_be_updated) / len(E)), flush=True)
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
                correctorsListT, KmsijT, MmsijT, BdmsijT, muTPrime = UpdateElements(tol, E, KmsijT, MmsijT, BdmsijT, correctorsListT, muTPrime) # tol scaled by maximal error indicator

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

            fHQuad = basis.T * MFineFEM * f

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

            Err = np.sqrt(np.dot((uSol - uLodFine).conj(), KFineFEM * (uSol - uLodFine)) + k**2*np.dot((uSol - uLodFine).conj(), MFineFEM * (uSol - uLodFine)))
            ErrEnergy = Err / np.sqrt(np.dot((uSol).conj(), KFineFEM * (uSol)) + k**2*np.dot((uSol).conj(), MFineFEM * (uSol)))
            print('---- ',np.abs(ErrEnergy), '\n***********************************************')
            
            # save errors in arrays
            relErrEnergy[counter-1,it] = ErrEnergy      
            
        print('\n')
        
    its = np.arange(1,maxit+1)
    plt.figure(1)
    plt.title('Relative energy errors w.r.t iterations for different tolerances - Ex 3')
    plt.plot(its,relErrEnergy[0,:],'x--',color='black', label='tol = 2')
    plt.plot(its,relErrEnergy[1,:],'x-',color='blue', label='tol = 1')
    plt.plot(its,relErrEnergy[2,:],'x-',color='green', label='tol = 0.5')
    plt.plot(its,relErrEnergy[3,:],'x-',color='orange', label='tol = 0.25')
    plt.plot(its,relErrEnergy[4,:],'x-',color='red', label='tol = 0.125')
    plt.plot(its,relErrEnergy[5,:],'x-',color='magenta', label='tol = 0.0625')
    plt.plot(its,relErrEnergy[6,:],'x--',color='black', label='tol = 0')
    plt.yscale('log')
    plt.legend()

    plt.show()
    
# run the code
helmholtz_nonlinear_adaptive(map,fineLevel,coarseLevel,maxIt) 
