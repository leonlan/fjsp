import docplex.cp.modeler as cpo
from docplex.cp.model import CpoExpr, CpoModel

from pyjobshop.ProblemData import Objective as DataObjective
from pyjobshop.ProblemData import ProblemData

from .Variables import Variables


class Objective:
    """
    Builds the objective expressions of the CP Optimizer model.
    """

    def __init__(
        self, model: CpoModel, data: ProblemData, variables: Variables
    ):
        self._model = model
        self._data = data
        self._task_vars = variables.task_vars
        self._job_vars = variables.job_vars

    def _makespan_expr(self) -> CpoExpr:
        """
        Returns an expression representing the makespan of the model.
        """
        return cpo.max(cpo.end_of(var, 0) for var in self._task_vars)

    def _tardy_jobs_expr(self) -> CpoExpr:
        """
        Returns an expression representing the number of tardy jobs.
        """
        exprs = []
        for job, var in zip(self._data.jobs, self._job_vars):
            is_tardy = cpo.greater(cpo.end_of(var) - job.due_date, 0)
            exprs.append(job.weight * is_tardy)

        return cpo.sum(exprs)  # type: ignore

    def _total_flow_time_expr(self) -> CpoExpr:
        """
        Returns an expression representing the total flow time of jobs.
        """
        total = []
        for job, var in zip(self._data.jobs, self._job_vars):
            flow_time = cpo.max(0, cpo.end_of(var) - job.release_date)
            total.append(job.weight * flow_time)

        return cpo.sum(total)  # type: ignore

    def _total_tardiness_expr(self) -> CpoExpr:
        """
        Returns an expression representing the total tardiness of jobs.
        """
        total = []
        for job, var in zip(self._data.jobs, self._job_vars):
            tardiness = cpo.max(0, cpo.end_of(var) - job.due_date)
            total.append(job.weight * tardiness)

        return cpo.sum(total)  # type: ignore

    def _total_earliness_expr(self) -> CpoExpr:
        """
        Returns an expression representing the total earliness of jobs.
        """
        total = []
        for job, var in zip(self._data.jobs, self._job_vars):
            earliness = cpo.max(0, job.due_date - cpo.end_of(var))
            total.append(job.weight * earliness)

        return cpo.sum(total)  # type: ignore

    def _max_tardiness_expr(self) -> CpoExpr:
        """
        Returns an expression representing the maximum tardiness of jobs.
        """
        tardiness = [
            job.weight * cpo.max(0, cpo.end_of(var) - job.due_date)
            for job, var in zip(self._data.jobs, self._job_vars)
        ]
        return cpo.max(tardiness)

    def _max_lateness_expr(self) -> CpoExpr:
        """
        Returns an expression representing the maximum lateness of jobs.
        """
        lateness = [
            job.weight * (cpo.end_of(var) - job.due_date)
            for job, var in zip(self._data.jobs, self._job_vars)
        ]
        return cpo.max(lateness)

    def _objective_expr(self, objective: DataObjective) -> CpoExpr:
        """
        Returns the expression corresponding to the given objective.
        """
        items = [
            (objective.weight_makespan, self._makespan_expr),
            (objective.weight_tardy_jobs, self._tardy_jobs_expr),
            (objective.weight_total_tardiness, self._total_tardiness_expr),
            (objective.weight_total_flow_time, self._total_flow_time_expr),
            (objective.weight_total_earliness, self._total_earliness_expr),
            (objective.weight_max_tardiness, self._max_tardiness_expr),
            (objective.weight_max_lateness, self._max_lateness_expr),
        ]
        exprs = [weight * expr() for weight, expr in items if weight > 0]
        return cpo.minimize(cpo.sum(exprs))

    def add_objective(self):
        """
        Adds the objective expression to the CP model.
        """
        obj_expr = self._objective_expr(self._data.objective)
        self._model.add(obj_expr)
