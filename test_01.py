import pyscipopt
from pyscipopt import Model
import ecole
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from geco.mips.miplib.base import Loader
from utility import instancetypes, generator_switcher, instancesizes
import sys
#
# def solve_prob(model):
#     MIP_model = model
#     MIP_model.optimize()
#
#     status = MIP_model.getStatus()
#     obj = MIP_model.getObjVal()
#     incumbent_solution = MIP_model.getBestSol()
#     n_vars = MIP_model.getNVars()
#     n_binvars = MIP_model.getNBinVars()
#     time = MIP_model.getSolvingTime()
#     n_sols = MIP_model.getNSols()
#
#     vars = MIP_model.getVars()
#
#     print('Status: ', status)
#     print('Solving time :', time)
#     print('obj:', obj)
#     print('number of solutions: ', n_sols)
#     print('Varibles: ', n_vars)
#     print('Binary vars: ', n_binvars)
#
#     n_supportbinvars = 0
#     for i in range(n_binvars):
#         val = MIP_model.getSolVal(incumbent_solution, vars[i])
#         # assert MIP_model.isFeasIntegral(val), "Error: Value of a binary varialbe is not integral!"
#         if MIP_model.isFeasEQ(val, 1.0):
#             n_supportbinvars += 1
#
#
#     print('Binary support: ', n_supportbinvars)
#     print('\n')
#
# instance_type = instancetypes[1]
# instance_size = instancesizes[0]
# dataset = instance_type + instance_size
#
# directory_opt = './result/generated_instances/' + instance_type + '/' + instance_size + '/'
# pathlib.Path(directory_opt).mkdir(parents=True, exist_ok=True)
#
# generator = generator_switcher(dataset)
# generator.seed(100)
# for i in range(5):
#     instance = next(generator)
#     MIP_model = instance.as_pyscipopt()
#
#     MIP_copy, MIP_copy_vars, success = MIP_model.createCopy(problemName='subMIPmodelCopy', origcopy=False)
#     MIP_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(problemName='subMIPmodelCopy', origcopy=False)
#
#     # MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
#     # MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
#     MIP_model.setParam('presolving/maxrounds', 0)
#     MIP_model.setParam('presolving/maxrestarts', 0)
#     # MIP_model.setParam("limits/nodes", 1)
#     MIP_model.setParam('limits/solutions', 1)
#
#     MIP_copy.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
#     # MIP_copy.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
#     MIP_copy.setParam('presolving/maxrounds', 0)
#     MIP_copy.setParam('presolving/maxrestarts', 0)
#     MIP_copy.setParam("limits/nodes", 1)
#     # MIP_copy.setParam("limits/gap", 10)
#
#     MIP_copy2.setParam('presolving/maxrounds', 0)
#     MIP_copy2.setParam('presolving/maxrestarts', 0)
#
#     solve_prob(MIP_model)
#     solve_prob(MIP_copy)
#     solve_prob(MIP_copy2)

a = False
b = False
c = a and b
print(c)
