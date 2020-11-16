# Modifications to GRIDLOD for nonlinear Helmholtz setup
# Copyright holders: Roland Maier, Barbara Verfuerth
# 2020

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import gc
import warnings

from gridlod import fem
from gridlod import util
from gridlod import linalg
from gridlod import coef
from gridlod import transport
from gridlod import interp
from gridlod import world

# Saddle point problem solver
class DirectSolver:
    def __init__(self):
        pass

    def solve(self, A, I, bList, fixed, NPatchCoarse=None, NCoarseElement=None):
        return saddleDirectComplex(A, I, bList, fixed)

def saddleDirectComplex(A, B, rhsList, fixed):
    A = linalg.imposeBoundaryConditionsStronglyOnMatrix(A, fixed) #
    rhsList = [linalg.imposeBoundaryConditionsStronglyOnVector(rhs, fixed) for rhs in rhsList] #
    B = linalg.imposeBoundaryConditionsStronglyOnInterpolation(B, fixed) #

    K = sparse.bmat([[A, B.T],
                     [B, None]], format='csc',dtype='complex128')

    xList = []
    for rhs in rhsList:
        b = np.zeros(K.shape[0],dtype='complex128')
        b[:np.size(rhs)] = rhs
        xAll = sparse.linalg.spsolve(K, b, use_umfpack=True)
        xList.append(xAll[:np.size(rhs)])
        
        if(np.any(np.absolute(B*xAll[:np.size(rhs)]) > 1e-10)):
            print(np.abs(B*xAll[:np.size(rhs)]))

    return xList

def ritzProjectionToFinePatch(patch,
                              APatchFull,
                              bPatchFullList,
                              IPatch,
                              saddleSolver=None):
    if saddleSolver is None:
        saddleSolver = DirectSolver()  # Fast for small patch problems

    world = patch.world
    d = np.size(patch.NPatchCoarse)
    NPatchFine = patch.NPatchFine
    NpFine = patch.NpFine

    # Find what patch faces are common to the world faces, and inherit
    # boundary conditions from the world for those. For the other
    # faces, all DoFs fixed (Dirichlet)
    boundaryMapWorld = world.boundaryConditions==0

    inherit0 = patch.iPatchWorldCoarse==0
    inherit1 = (patch.iPatchWorldCoarse + patch.NPatchCoarse)==world.NWorldCoarse

    boundaryMap = np.ones([d, 2], dtype='bool')
    boundaryMap[inherit0,0] = boundaryMapWorld[inherit0,0]
    boundaryMap[inherit1,1] = boundaryMapWorld[inherit1,1]

    # Using schur complement solver for the case when there are no
    # Dirichlet conditions does not work. Fix if necessary.

    fixed = util.boundarypIndexMap(NPatchFine, boundaryMap)
    
    projectionsList = saddleSolver.solve(APatchFull, IPatch, bPatchFullList, fixed, patch.NPatchCoarse, world.NCoarseElement)

    return projectionsList

class CoarseScaleInformation_helmholtz: 
    def __init__(self, Kij, Kmsij, muTPrime, Mij, Mmsij, Bdij, Bdmsij):
        self.Kij = Kij
        self.Kmsij = Kmsij
        self.muTPrime = muTPrime
        self.Mij = Mij
        self.Mmsij = Mmsij
        self.Bdij = Bdij
        self.Bdmsij = Bdmsij


def computeElementCorrector_helmholtz(patch, IPatch, aPatch, kPatch, k2Patch, ARhsList=None, MRhsList=None, saddleSolver=None):
    '''Compute the fine correctors over a patch.

    Compute the correctors

    B( Q_T_j, vf)_{U_K(T)} = B( ARhs_j, vf)_{T} + (MRhs_j, vf)_{T}

    where B is the sesquilinear form associated with the linear Helmholtz eq.
    '''

    while callable(IPatch):
        IPatch = IPatch()

    while callable(aPatch):
        aPatch = aPatch()

    while callable(kPatch):
        kPatch = kPatch()

    while callable(k2Patch):
        k2Patch = k2Patch()


    assert(ARhsList is not None or MRhsList is not None)
    numRhs = None

    if ARhsList is not None:
        assert(numRhs is None or numRhs == len(ARhsList))
        numRhs = len(ARhsList)

    if MRhsList is not None:
        assert(numRhs is None or numRhs == len(MRhsList))
        numRhs = len(MRhsList)

    world = patch.world
    NCoarseElement = world.NCoarseElement
    NPatchCoarse = patch.NPatchCoarse
    d = np.size(NCoarseElement)

    NPatchFine = NPatchCoarse*NCoarseElement
    NtFine = np.prod(NPatchFine)
    NpFineCoarseElement = np.prod(NCoarseElement+1)
    NpCoarse = np.prod(NPatchCoarse+1)
    NpFine = np.prod(NPatchFine+1)

    assert(aPatch.shape[0] == NtFine)
    assert(aPatch.ndim == 1 or aPatch.ndim == 3)
    assert(kPatch.ndim == 1)
    assert(k2Patch.ndim == 1)

    if aPatch.ndim == 1:
        ALocFine = world.ALocFine
    elif aPatch.ndim == 3:
        ALocFine = world.ALocMatrixFine

    MLocFine = world.MLocFine
    BdLocFine = fem.localBoundaryMassMatrixGetter(NCoarseElement*world.NWorldCoarse)

    iElementPatchCoarse = patch.iElementPatchCoarse
    elementFinetIndexMap = util.extractElementFine(NPatchCoarse,
                                                   NCoarseElement,
                                                   iElementPatchCoarse,
                                                   extractElements=True)
    elementFinepIndexMap = util.extractElementFine(NPatchCoarse,
                                                   NCoarseElement,
                                                   iElementPatchCoarse,
                                                   extractElements=False)

    # global boundary?
    bdMapWorld = world.boundaryConditions == 1

    # on element
    bdMapElement = np.zeros([d, 2], dtype='bool')

    inheritElement0 = patch.iElementWorldCoarse == 0
    inheritElement1 = (patch.iElementWorldCoarse + np.ones(d)) == world.NWorldCoarse

    bdMapElement[inheritElement0, 0] = bdMapWorld[inheritElement0, 0]
    bdMapElement[inheritElement1, 1] = bdMapWorld[inheritElement1, 1]

    # on patch
    inherit0 = patch.iPatchWorldCoarse == 0
    inherit1 = (patch.iPatchWorldCoarse + NPatchCoarse) == world.NWorldCoarse

    bdMapPatch = np.zeros([d, 2], dtype='bool')
    bdMapPatch[inherit0, 0] = bdMapWorld[inherit0, 0]
    bdMapPatch[inherit1, 1] = bdMapWorld[inherit1, 1]

    if ARhsList is not None:
        AElementFull = fem.assemblePatchMatrix(NCoarseElement, ALocFine, aPatch[elementFinetIndexMap])
        k2MElementFull = fem.assemblePatchMatrix(NCoarseElement, MLocFine, k2Patch[elementFinetIndexMap])
        kBdElementFull = fem.assemblePatchBoundaryMatrix(NCoarseElement, BdLocFine, kPatch[elementFinetIndexMap],bdMapElement)
    if MRhsList is not None:
        MElementFull = fem.assemblePatchMatrix(NCoarseElement, MLocFine)
    APatchFull = fem.assemblePatchMatrix(NPatchFine, ALocFine, aPatch)
    k2MPatchFull = fem.assemblePatchMatrix(NPatchFine, MLocFine, k2Patch)
    kBdPatchFull = fem.assemblePatchBoundaryMatrix(NPatchFine, BdLocFine, kPatch, bdMapPatch)

    SPatchFull = APatchFull - k2MPatchFull + 1j*kBdPatchFull

    bPatchFullList = []
    for rhsIndex in range(numRhs):
        bPatchFull = np.zeros(NpFine,dtype='complex128')
        if ARhsList is not None:
            bPatchFull[elementFinepIndexMap] += (AElementFull - k2MElementFull + 1j*kBdElementFull)*ARhsList[rhsIndex]
        if MRhsList is not None:
            bPatchFull[elementFinepIndexMap] += MElementFull*MRhsList[rhsIndex]
        bPatchFullList.append(bPatchFull)

    correctorsList = ritzProjectionToFinePatch(patch,
                                               SPatchFull,
                                               bPatchFullList,
                                               IPatch,
                                               saddleSolver)

    return correctorsList

def computeBasisCorrectors_helmholtz(patch, IPatch, aPatch, kPatch, k2Patch, saddleSolver=None):
    '''Compute the fine basis correctors over the patch.

    Compute the correctors Q_T\lambda_i:

    B( Q_T lambda_j, vf)_{U_K(T)} = B( lambda_j, vf)_{T}

    where B is the sesquilinear form associated with the linear Helmholtz eq.
    '''

    ARhsList = list(patch.world.localBasis.T)

    return computeElementCorrector_helmholtz(patch, IPatch, aPatch, kPatch, k2Patch, ARhsList, saddleSolver=None)




def computeEftErrorIndicatorsCoarseFromGreeks(etaT, cetaTPrime, greeksPatch):
    '''Compute the coarse error idicator E(T) from the "greeks" delta and kappa,
    where

    deltaMaxTPrime = || ANew^{-1/2} (ANew - AOld) AOld^{-1/2} ||_max(TPrime)
                                                     over all coarse elements TPrime in the patch

    kappaMaxT = || AOld^{1/2) ANew^{-1/2} ||_max(T)
                                 over the patch coarse T (patch.iPatchWorldCoarse)

    nuT = || f - f_ref ||_L2(T)

    '''

    while callable(greeksPatch):
        greeksPatch = greeksPatch()

    deltaMaxTPrime, kappaMaxT, xiMaxT, nuT, gammaT = greeksPatch

    eft_square = xiMaxT**2 * nuT**2 * etaT
    eRft_square = kappaMaxT**2 * np.sum((deltaMaxTPrime**2)*cetaTPrime)
    return np.sqrt(eft_square), np.sqrt(eRft_square)



def computeErrorIndicatorCoarse_helmholtz(patch, muTPrime, aPatchOld, aPatchNew):
    ''' Compute the coarse error idicator E(T) with explicit value of AOld and ANew.

    This requires muTPrime from CSI and the new and old coefficient.
    '''

    while callable(muTPrime):
        muTPrime = muTPrime()

    while callable(aPatchOld):
        aPatchOld = aPatchOld()

    while callable(aPatchNew):
        aPatchNew = aPatchNew()

    aOld = aPatchOld
    aNew = aPatchNew

    world = patch.world
    NPatchCoarse = patch.NPatchCoarse
    NCoarseElement = world.NCoarseElement
    NPatchFine = NPatchCoarse*NCoarseElement
    iElementPatchCoarse = patch.iElementPatchCoarse

    elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse-1, iElementPatchCoarse)

    TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
    TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

    TPrimeIndices = np.add.outer(TPrimeFinetStartIndices, TPrimeFinetIndexMap)
    aTPrime = aNew[TPrimeIndices]
    aOldTPrime = aOld[TPrimeIndices]

    deltaMaxTPrime = np.max(np.abs(aTPrime - aOldTPrime), axis=1)

    epsilonTSquare = np.sum((deltaMaxTPrime ** 2) * muTPrime)

    return np.sqrt(epsilonTSquare)


def performTPrimeLoop_helmholtz(patch, lambdasList, correctorsList, aPatch, kPatch, k2Patch, accumulate):

    while callable(aPatch):
        aPatch = aPatch()

    while callable(kPatch):
        kPatch = kPatch()

    while callable(k2Patch):
        k2Patch = k2Patch()

    world = patch.world
    NCoarseElement = world.NCoarseElement
    NPatchCoarse = patch.NPatchCoarse
    NPatchFine = NPatchCoarse*NCoarseElement

    NTPrime = np.prod(NPatchCoarse)
    NpPatchCoarse = np.prod(NPatchCoarse+1)

    d = np.size(NPatchCoarse)

    assert(aPatch.ndim == 1 or aPatch.ndim == 3)
    assert(kPatch.ndim == 1)
    assert(k2Patch.ndim == 1)

    if aPatch.ndim == 1:
        ALocFine = world.ALocFine
    elif aPatch.ndim == 3:
        ALocFine = world.ALocMatrixFine

    MLocFine = world.MLocFine
    BdLocFine = fem.localBoundaryMassMatrixGetter(NCoarseElement*world.NWorldCoarse)

    lambdas = np.column_stack(lambdasList)
    numLambdas = len(lambdasList)
    
    
    TPrimeCoarsepStartIndices = util.lowerLeftpIndexMap(NPatchCoarse-1, NPatchCoarse)
    TPrimeCoarsepIndexMap = util.lowerLeftpIndexMap(np.ones_like(NPatchCoarse), NPatchCoarse)

    TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
    TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

    TPrimeFinepStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine, NCoarseElement)
    TPrimeFinepIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)

    TInd = util.convertpCoordIndexToLinearIndex(NPatchCoarse-1, patch.iElementPatchCoarse)

    QPatch = np.column_stack(correctorsList)

    # global boundary
    bdMapWorld = world.boundaryConditions == 1

    for (TPrimeInd,
         TPrimeCoarsepStartIndex,
         TPrimeFinetStartIndex,
         TPrimeFinepStartIndex) \
         in zip(np.arange(NTPrime),
                TPrimeCoarsepStartIndices,
                TPrimeFinetStartIndices,
                TPrimeFinepStartIndices):

        aTPrime = aPatch[TPrimeFinetStartIndex + TPrimeFinetIndexMap]
        kTPrime = kPatch[TPrimeFinetStartIndex + TPrimeFinetIndexMap]
        k2TPrime = k2Patch[TPrimeFinetStartIndex + TPrimeFinetIndexMap]
        KTPrime = fem.assemblePatchMatrix(NCoarseElement, ALocFine, aTPrime)
        MTPrime = fem.assemblePatchMatrix(NCoarseElement, MLocFine, k2TPrime)
        L2TPrime = fem.assemblePatchMatrix(NCoarseElement, MLocFine)

        # boundary on element
        bdMapElement = np.zeros([d, 2], dtype='bool')

        iElementTPrime = patch.iPatchWorldCoarse + util.convertpLinearIndexToCoordIndex(NPatchCoarse-1,TPrimeInd)

        inheritElement0 =  iElementTPrime == 0
        inheritElement1 = (iElementTPrime + np.ones(d)) == world.NWorldCoarse

        bdMapElement[inheritElement0, 0] = bdMapWorld[inheritElement0, 0]
        bdMapElement[inheritElement1, 1] = bdMapWorld[inheritElement1, 1]

        BdTPrime = fem.assemblePatchBoundaryMatrix(NCoarseElement, BdLocFine, kTPrime, bdMapElement) 
        P = lambdas
        Q = QPatch[TPrimeFinepStartIndex + TPrimeFinepIndexMap,:]
        _KTPrimeij = np.dot(P.T, KTPrime*Q)
        TPrimei = TPrimeCoarsepStartIndex + TPrimeCoarsepIndexMap

        _MTPrimeij = np.dot(P.T, MTPrime*Q)
        _BdTPrimeij = np.dot(P.T, BdTPrime*Q)

        CTPrimeij = np.dot(Q.T, L2TPrime*Q)
        BTPrimeij = np.dot(P.T, L2TPrime*Q)

        accumulate(TPrimeInd, TPrimei, P, Q, KTPrime, _KTPrimeij, MTPrime, BdTPrime,
                   _MTPrimeij, _BdTPrimeij, L2TPrime, CTPrimeij, BTPrimeij)
    
def computeBasisCoarseQuantities_helmholtz(patch, correctorsList, aPatch, kPatch, k2Patch):
    ''' Compute the coarse quantities for the basis and its correctors

    Compute the tensors (T is implicit by the patch definition) and
    return them in a CoarseScaleInformation object:

       Kij   = (A \nabla lambda_j, \nabla lambda_i)_{T}
       KmsTij = (A \nabla (lambda_j - corrector_j), \nabla lambda_i)_{U_k(T)}
       Mij = (k^2n lambda_j, lambda_i)_{T}
       Mmsij = (k^2n (lambda_j - corrector_j), lambda_i)_{U_k(T)}
       Bdij = (k lambda_j, lambda_i)_{\partial T \cap \partial \Omega}
       Bdmsij = (k (lambda_j - corrector_j), lambda_i)_{\partial T \cap \partial \Omega}
       muTT'  = max_{\alpha_j} || \alpha_j (\chi_T lambda_j - corrector_j) ||^2_T' / || \alpha_j lambda_j ||^2_T

       Auxiliary quantities are computed, but not saved
    '''

    lambdasList = list(patch.world.localBasis.T)

    NPatchCoarse = patch.NPatchCoarse
    NTPrime = np.prod(NPatchCoarse)
    NpPatchCoarse = np.prod(NPatchCoarse+1)
    numLambdas = len(lambdasList)

    TInd = util.convertpCoordIndexToLinearIndex(patch.NPatchCoarse-1, patch.iElementPatchCoarse)
    
    Kmsij = np.zeros((NpPatchCoarse, numLambdas), dtype='complex128')
    Mmsij = np.zeros((NpPatchCoarse, numLambdas), dtype='complex128')
    Bdmsij = np.zeros((NpPatchCoarse, numLambdas), dtype='complex128')
    LTPrimeij = np.zeros((NTPrime, numLambdas, numLambdas), dtype='complex128')
    Kij = np.zeros((numLambdas, numLambdas), dtype='complex128')
    Mij = np.zeros((numLambdas, numLambdas), dtype='complex128')
    Bdij = np.zeros((numLambdas, numLambdas), dtype='complex128')
    L2ij = np.zeros((numLambdas, numLambdas), dtype='complex128')

    def accumulate(TPrimeInd, TPrimei, P, Q, KTPrime, _KTPrimeij, MTPrime, BdTPrime,
                   _MTPrimeij, _BdTPrimeij, L2TPrime, CTPrimeij, BTPrimeij):
        if TPrimeInd == TInd:
            Kij[:] = np.dot(P.T, KTPrime*P)
            Mij[:] = np.dot(P.T, MTPrime*P)
            Bdij[:] = np.dot(P.T, BdTPrime*P)
            L2ij[:] = np.dot(P.T, L2TPrime*P)
            LTPrimeij[TPrimeInd] = CTPrimeij \
                                   - BTPrimeij \
                                   - BTPrimeij.T \
                                   + L2ij

            Kmsij[TPrimei,:] += Kij - _KTPrimeij
            Mmsij[TPrimei,:] += Mij - _MTPrimeij
            Bdmsij[TPrimei,:] += Bdij - _BdTPrimeij
        else:
            LTPrimeij[TPrimeInd] = CTPrimeij
            Kmsij[TPrimei,:] += -_KTPrimeij
            Mmsij[TPrimei,:] += -_MTPrimeij
            Bdmsij[TPrimei,:] += -_BdTPrimeij

    performTPrimeLoop_helmholtz(patch, lambdasList, correctorsList, aPatch, kPatch, k2Patch, accumulate)

    
    muTPrime = np.zeros(NTPrime, dtype='complex128')
    cutRows = 0
    while np.linalg.cond(L2ij[cutRows:,cutRows:]) > 1e8:
        cutRows = cutRows + 1
    for TPrimeInd in np.arange(NTPrime):
        # Solve eigenvalue problem LTPrimeij x = mu_TPrime Mij x
        eigenvalues = scipy.linalg.eigvals(LTPrimeij[TPrimeInd][cutRows:,cutRows:], L2ij[cutRows:,cutRows:])
        muTPrime[TPrimeInd] = np.max(np.real(eigenvalues))

    return CoarseScaleInformation_helmholtz(Kij, Kmsij, muTPrime, Mij, Mmsij, Bdij, Bdmsij)
