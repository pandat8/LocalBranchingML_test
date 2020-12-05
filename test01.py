from pyscipopt import Model

import ecole

"""Test output of set cover instance."""
instances = ecole.instance.SetCoverGenerator()
instance = next(instances)
pyscipopt_model = instance.as_pyscipopt()
pyscipopt_model.optimize()

status = pyscipopt_model.getStatus()
print("* Model status: %s" % status)
if status == 'optimal':
    print("Optimal value:", pyscipopt_model.getObjVal())
    # for v in pyscipopt_model.getVars():
    #     if v.name != "n":
    #         print("%s: %d" % (v, pyscipopt_model.getVal(v)))
else:
    print("* No variable is printed if model status is not optimal")

model = Model("MILP")

x = model.addVar("x")
y = model.addVar("y", vtype="INTEGER")
model.setObjective(x + y)
model.addCons(2*x - y*y >= 0)
model.optimize()
sol = model.getBestSol()
print("Model MILP is solved ")
print("x: {}".format(sol[x]))
print("y: {}".format(sol[y]))




