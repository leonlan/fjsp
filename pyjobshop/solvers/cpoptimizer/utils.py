import docplex.cp.modeler as cpo
from docplex.cp.expression import CpoExpr, CpoIntervalVar


def presence_of(var: CpoIntervalVar) -> CpoExpr:
    """
    Wrapper around CP Optimizer's ``presence_of`` function to return an
    interval variable's presence status. If the variable is always present,
    calling ``cpo.presence_of`` emits warnings, so instead we return 1.
    """
    return cpo.presence_of(var) if var.is_optional() else 1
