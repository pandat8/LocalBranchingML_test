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
    for v in pyscipopt_model.getVars():
        if v.name != "n":
            print("%s: %d" % (v, pyscipopt_model.getVal(v)))
else:
    print("* No variable is printed if model status is not optimal")


