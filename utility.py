import ecole
import numpy as np

instancetypes = ['setcovering', 'capacitedfacility', 'independentset', 'combinatorialauction'
                 ]
instancesizes = ['-small','-mid','-large']
lbconstraint_modes = ['symmetric', 'asymmetric']
incumbent_modes = ['firstsol', 'rootsol']

def generator_switcher(dataset):
    switcher = {
        instancetypes[0] + instancesizes[0]: lambda: ecole.instance.SetCoverGenerator(n_rows=5000, n_cols=2000, density=0.01),
        instancetypes[1] + instancesizes[0]: lambda: ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=30,
                                                                                                         n_facilities=200,
                                                                                                         continuous_assignment=True,
                                                                                                         capacity_interval=(5,10)),
        # instancetypes[1] + instancesizes[0]: lambda: ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=50, n_facilities=100, continuous_assignment=True, capacity_interval=[5,10]),
        instancetypes[2] + instancesizes[0]: lambda: ecole.instance.IndependentSetGenerator(n_nodes=1000),
        instancetypes[3] + instancesizes[0]: lambda: ecole.instance.CombinatorialAuctionGenerator(n_items=300, n_bids=300),
        instancetypes[0] + instancesizes[1]: lambda: ecole.instance.SetCoverGenerator(n_rows=10000, n_cols=4000, density=0.01),
        instancetypes[1] + instancesizes[1]: lambda: ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=200, continuous_assignment=True, capacity_interval=[5,10]),
        instancetypes[2] + instancesizes[1]: lambda: ecole.instance.IndependentSetGenerator(n_nodes=2000),
        instancetypes[3] + instancesizes[1]: lambda: ecole.instance.CombinatorialAuctionGenerator(n_items=600, n_bids=600),
    }
    return switcher.get(dataset, lambda : "invalide argument")()

def copy_sol(mip_original, mip_target, sol, mip_target_vars):
    """
    copy the sol of mip original to mip target(copy of mip original)
    :param mip_original:
    :param mip_target:
    :param sol:
    :param mip_target_vars:
    :return:
    """

    sol_mip_target = mip_target.createSol()

    # create a primal solution for the copy MIP by copying the solution of original MIP
    n_vars = mip_original.getNVars()
    mip_original_vars = mip_original.getVars()
    # print("Number of variables Original: ", n_vars)
    # print("Number of variables Copy", mip_target.getNVars())
    # print("Number of bin variables Copy", mip_target.getNBinVars())
    for j in range(n_vars):
        val = mip_original.getSolVal(sol, mip_original_vars[j])
        mip_target.setSolVal(sol_mip_target, mip_target_vars[j], val)
    feasible = mip_target.checkSol(solution=sol_mip_target)

    if feasible:
        mip_target.addSol(sol_mip_target, False)
        # print("the feasible solution of " + mip_target.getProbName() + " is added to target mid")
    else:
        print("Error: the trivial solution of " + mip_target.getProbName() + " is not feasible!")
    return mip_target, sol_mip_target

def binary_support(mip, sol):
    n_binvars = mip.getNBinVars()
    vars = mip.getVars()
    n_supportbinvars = 0
    for i in range(n_binvars):
        val = mip.getSolVal(sol, vars[i])
        assert mip.isFeasIntegral(val), "Error: Value of a binary varialbe is not integral!"
        if mip.isFeasEQ(val, 1.0):
            n_supportbinvars += 1
    return n_supportbinvars

def mean_filter(a, kernal_size):
    """
    mean filter of a
    :param a:
    :param kernal_size:
    :return:
    """
    a_mean = np.zeros(a.shape)

    k = int((kernal_size -1)/2)

    for i in range(a.shape[0]):
        if i < k or i > (a.shape[0] - 1 -k):
            a_mean[i] = a[i]
        else:
            for n in range(kernal_size):
                a_mean[i] +=  a[i-k+n]
            a_mean[i] = a_mean[i] / kernal_size
    return a_mean


def mean_forward_filter(a, kernal_size):
    a_mean = np.zeros(a.shape)

    k = kernal_size

    for i in range(a.shape[0]):
        if i > (a.shape[0] - 1 - k):
            a_mean[i] = a[i]
        else:
            for n in range(kernal_size):
                a_mean[i] += a[i + n]
            a_mean[i] = a_mean[i] / kernal_size
    return a_mean

