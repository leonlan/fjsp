from typing import Optional

from docplex.cp.model import CpoModel
from docplex.cp.solution import CpoSolveResult

from pyjobshop.ProblemData import ProblemData
from pyjobshop.Result import Result, SolveStatus
from pyjobshop.Solution import Solution, TaskData

from .Constraints import Constraints
from .Objective import Objective
from .Variables import Variables


class Solver:
    """
    Wrapper around the CP Optimizer CP model.

    Parameters
    ----------
    data
        The problem data instance.
    """

    def __init__(self, data: ProblemData):
        self._data = data

        self._model = CpoModel()
        self._variables = Variables(self._model, data)
        self._constraints = Constraints(self._model, data, self._variables)
        self._objective = Objective(self._model, data, self._variables)

        self._constraints.add_constraints()
        self._objective.add_objective()

    def _get_solve_status(self, status: str) -> SolveStatus:
        if status == "Optimal":
            return SolveStatus.OPTIMAL
        elif status == "Feasible":
            return SolveStatus.FEASIBLE
        elif status == "Infeasible":
            return SolveStatus.INFEASIBLE
        else:
            return SolveStatus.TIME_LIMIT

    def _convert_to_solution(self, result: CpoSolveResult) -> Solution:
        """
        Converts an CpoSolveResult object to a solution.
        """
        tasks = {}

        for var in result.get_all_var_solutions():  # type: ignore
            name = var.get_name()

            # Scheduled tasks are inferred from present mode variables.
            if name.startswith("M") and var.is_present():
                mode, task = map(int, name[1:].split("_"))
                resources = self._data.modes[mode].resources
                start = var.start
                end = var.end
                tasks[task] = TaskData(
                    mode, resources, start, end, present=True
                )

        for idx in range(self._data.num_tasks):
            if idx not in tasks:
                tasks[idx] = TaskData(idx, [], 0, 0, present=False)

        return Solution([tasks[idx] for idx in range(self._data.num_tasks)])

    def solve(
        self,
        time_limit: float = float("inf"),
        display: bool = False,
        num_workers: Optional[int] = None,
        initial_solution: Optional[Solution] = None,
        **kwargs,
    ) -> Result:
        """
        Solves the given problem data instance with Google OR-Tools.

        Parameters
        ----------
        time_limit
            The time limit for the solver in seconds.
        display
            Whether to display the solver output. Default ``False``.
        num_workers
            The number of workers to use for parallel solving. If not set, all
            available CPU cores are used.
        initial_solution
            Initial solution to start the solver from. Default is no solution.
        kwargs
            Additional parameters passed to the solver.

        Returns
        -------
        Result
            A Result object containing the best found solution and additional
            information about the solver run.
        """
        if initial_solution is not None:
            self._variables.warmstart(initial_solution)

        params = {
            "TimeLimit": time_limit,
            "LogVerbosity": "Terse" if display else "Quiet",
            "Workers": num_workers if num_workers is not None else "Auto",
        }
        params.update(kwargs)  # this will override existing parameters!

        cp_result: CpoSolveResult = self._model.solve(**params)  # type: ignore
        status = cp_result.get_solve_status()

        if status in ["Optimal", "Feasible"]:
            solution = self._convert_to_solution(cp_result)
            objective: float = cp_result.get_objective_value()  # type: ignore
        else:
            # No feasible solution due to infeasible instance or time limit.
            solution = Solution([])
            objective = float("inf")

        return Result(
            objective=objective,
            lower_bound=cp_result.get_objective_bound(),
            status=self._get_solve_status(status),
            runtime=cp_result.get_solve_time(),
            best=solution,
        )
