from typing import Optional, Sequence, Union

from pyjobshop.constants import MAX_VALUE
from pyjobshop.ProblemData import (
    Consecutive,
    Constraints,
    DifferentResources,
    EndBeforeEnd,
    EndBeforeStart,
    IdenticalResources,
    IfThen,
    Job,
    Machine,
    Mode,
    NonRenewable,
    Objective,
    ProblemData,
    Renewable,
    Resource,
    SetupTime,
    StartBeforeEnd,
    StartBeforeStart,
    Task,
)
from pyjobshop.Result import Result
from pyjobshop.Solution import Solution
from pyjobshop.solve import solve


class Model:
    """
    A simple interface for building a scheduling problem step-by-step.
    """

    def __init__(self):
        self._jobs: list[Job] = []
        self._resources: list[Resource] = []
        self._tasks: list[Task] = []
        self._modes: list[Mode] = []
        self._constraints = Constraints()
        self._objective: Objective = Objective(weight_makespan=1)

        self._id2job: dict[int, int] = {}
        self._id2resource: dict[int, int] = {}
        self._id2task: dict[int, int] = {}

    @property
    def jobs(self) -> list[Job]:
        """
        Returns the list of jobs in the model.
        """
        return self._jobs

    @property
    def resources(self) -> list[Resource]:
        """
        Returns the list of resources in the model.
        """
        return self._resources

    @property
    def tasks(self) -> list[Task]:
        """
        Returns the list of tasks in the model.
        """
        return self._tasks

    @property
    def modes(self) -> list[Mode]:
        """
        Returns the list of modes in the model.
        """
        return self._modes

    @property
    def constraints(self) -> Constraints:
        """
        Returns the constraints in this model.
        """
        return self._constraints

    @property
    def objective(self) -> Objective:
        """
        Returns the objective function in this model.
        """
        return self._objective

    @classmethod
    def from_data(cls, data: ProblemData):
        """
        Creates a Model instance from a ProblemData instance.
        """
        model = cls()

        for job in data.jobs:
            model.add_job(
                weight=job.weight,
                release_date=job.release_date,
                deadline=job.deadline,
                due_date=job.due_date,
                name=job.name,
            )

        for resource in data.resources:
            if isinstance(resource, Machine):
                model.add_machine(name=resource.name)
            elif isinstance(resource, Renewable):
                model.add_renewable(
                    capacity=resource.capacity,
                    name=resource.name,
                )
            elif isinstance(resource, NonRenewable):
                model.add_non_renewable(
                    capacity=resource.capacity,
                    name=resource.name,
                )
            else:
                raise ValueError(f"Unknown resource type: {type(resource)}")

        for task in data.tasks:
            model.add_task(
                job=model.jobs[task.job] if task.job is not None else None,
                earliest_start=task.earliest_start,
                latest_start=task.latest_start,
                earliest_end=task.earliest_end,
                latest_end=task.latest_end,
                fixed_duration=task.fixed_duration,
                optional=task.optional,
                name=task.name,
            )

        for mode in data.modes:
            model.add_mode(
                task=model.tasks[mode.task],
                resources=[model.resources[res] for res in mode.resources],
                duration=mode.duration,
                demands=mode.demands,
            )

        tasks = model.tasks

        for idx1, idx2, delay in data.constraints.start_before_start:
            model.add_start_before_start(tasks[idx1], tasks[idx2], delay)

        for idx1, idx2, delay in data.constraints.start_before_end:
            model.add_start_before_end(tasks[idx1], tasks[idx2], delay)

        for idx1, idx2, delay in data.constraints.end_before_start:
            model.add_end_before_start(tasks[idx1], tasks[idx2], delay)

        for idx1, idx2, delay in data.constraints.end_before_end:
            model.add_end_before_end(tasks[idx1], tasks[idx2], delay)

        for idx1, idx2 in data.constraints.identical_resources:
            model.add_identical_resources(tasks[idx1], tasks[idx2])

        for idx1, idx2 in data.constraints.different_resources:
            model.add_different_resources(tasks[idx1], tasks[idx2])

        for idx1, idcs2 in data.constraints.if_then:
            model.add_if_then(tasks[idx1], [tasks[idx2] for idx2 in idcs2])

        for idx1, idx2, res_idx in data.constraints.consecutive:
            model.add_consecutive(
                tasks[idx1],
                tasks[idx2],
                model.resources[res_idx],  # type: ignore
            )

        for res_idx, idx1, idx2, duration in data.constraints.setup_times:
            model.add_setup_time(
                machine=model.resources[res_idx],  # type: ignore
                task1=tasks[idx1],
                task2=tasks[idx2],
                duration=duration,
            )

        model.set_objective(
            weight_makespan=data.objective.weight_makespan,
            weight_tardy_jobs=data.objective.weight_tardy_jobs,
            weight_total_tardiness=data.objective.weight_total_tardiness,
            weight_total_flow_time=data.objective.weight_total_flow_time,
            weight_total_earliness=data.objective.weight_total_earliness,
            weight_max_tardiness=data.objective.weight_max_tardiness,
            weight_max_lateness=data.objective.weight_max_lateness,
        )

        return model

    def data(self) -> ProblemData:
        """
        Returns a ProblemData object containing the problem instance.
        """
        return ProblemData(
            jobs=self.jobs,
            resources=self.resources,
            tasks=self.tasks,
            modes=self.modes,
            constraints=self.constraints,
            objective=self.objective,
        )

    def add_job(
        self,
        weight: int = 1,
        release_date: int = 0,
        deadline: int = MAX_VALUE,
        due_date: Optional[int] = None,
        name: str = "",
    ) -> Job:
        """
        Adds a job to the model.
        """
        job = Job(weight, release_date, deadline, due_date, name=name)

        self._id2job[id(job)] = len(self.jobs)
        self._jobs.append(job)

        return job

    def add_machine(self, name: str = "") -> Machine:
        """
        Adds a machine to the model.
        """
        machine = Machine(name=name)

        self._id2resource[id(machine)] = len(self.resources)
        self._resources.append(machine)

        return machine

    def add_renewable(self, capacity: int, name: str = "") -> Renewable:
        """
        Adds a renewable resource to the model.
        """
        resource = Renewable(capacity=capacity, name=name)

        self._id2resource[id(resource)] = len(self.resources)
        self._resources.append(resource)

        return resource

    def add_non_renewable(self, capacity: int, name: str = "") -> NonRenewable:
        """
        Adds a non-renewable resource to the model.
        """
        resource = NonRenewable(capacity=capacity, name=name)

        self._id2resource[id(resource)] = len(self.resources)
        self._resources.append(resource)

        return resource

    def add_task(
        self,
        job: Optional[Job] = None,
        earliest_start: int = 0,
        latest_start: int = MAX_VALUE,
        earliest_end: int = 0,
        latest_end: int = MAX_VALUE,
        fixed_duration: bool = True,
        optional: bool = False,
        name: str = "",
    ) -> Task:
        """
        Adds a task to the model.
        """
        job_idx = self._id2job[id(job)] if job is not None else None
        task = Task(
            job_idx,
            earliest_start,
            latest_start,
            earliest_end,
            latest_end,
            fixed_duration,
            optional,
            name,
        )

        task_idx = len(self.tasks)
        self._id2task[id(task)] = task_idx
        self._tasks.append(task)

        if job_idx is not None:
            self._jobs[job_idx].add_task(task_idx)

        return task

    def add_mode(
        self,
        task: Task,
        resources: Union[Resource, Sequence[Resource]],
        duration: int,
        demands: Optional[Union[int, list[int]]] = None,
    ) -> Mode:
        """
        Adds a processing mode to the model.
        """
        if isinstance(resources, (Machine, Renewable, NonRenewable)):
            resources = [resources]

        if isinstance(demands, int):
            demands = [demands]

        task_idx = self._id2task[id(task)]
        resource_idcs = [self._id2resource[id(res)] for res in resources]
        mode = Mode(task_idx, resource_idcs, duration, demands)
        self._modes.append(mode)

        return mode

    def add_start_before_start(
        self, task1: Task, task2: Task, delay: int = 0
    ) -> StartBeforeStart:
        """
        Adds a constraint that task 1 must start before task 2 starts, with an
        optional delay.
        """
        idx1, idx2 = self._id2task[id(task1)], self._id2task[id(task2)]
        constraint = StartBeforeStart(idx1, idx2, delay)
        self._constraints.start_before_start.append(constraint)

        return constraint

    def add_start_before_end(
        self, task1: Task, task2: Task, delay: int = 0
    ) -> StartBeforeEnd:
        """
        Adds a constraint that task 1 must start before task 2 ends, with an
        optional delay.
        """
        idx1, idx2 = self._id2task[id(task1)], self._id2task[id(task2)]
        constraint = StartBeforeEnd(idx1, idx2, delay)
        self._constraints.start_before_end.append(constraint)

        return constraint

    def add_end_before_start(
        self, task1: Task, task2: Task, delay: int = 0
    ) -> EndBeforeStart:
        """
        Adds a constraint that task 1 must end before task 2 starts, with an
        optional delay.
        """
        idx1, idx2 = self._id2task[id(task1)], self._id2task[id(task2)]
        constraint = EndBeforeStart(idx1, idx2, delay)
        self._constraints.end_before_start.append(constraint)

        return constraint

    def add_end_before_end(
        self, task1: Task, task2: Task, delay: int = 0
    ) -> EndBeforeEnd:
        """
        Adds a constraint that task 1 must end before task 2 ends, with an
        optional delay.
        """
        idx1, idx2 = self._id2task[id(task1)], self._id2task[id(task2)]
        constraint = EndBeforeEnd(idx1, idx2, delay)
        self._constraints.end_before_end.append(constraint)

        return constraint

    def add_identical_resources(
        self, task1: Task, task2: Task
    ) -> IdenticalResources:
        """
        Adds a constraint that two tasks must be scheduled with modes that
        require identical resources.
        """
        idx1, idx2 = self._id2task[id(task1)], self._id2task[id(task2)]
        constraint = IdenticalResources(idx1, idx2)
        self._constraints.identical_resources.append(constraint)

        return constraint

    def add_different_resources(
        self, task1: Task, task2: Task
    ) -> DifferentResources:
        """
        Adds a constraint that the two tasks must be scheduled with modes that
        require different resources.
        """
        idx1, idx2 = self._id2task[id(task1)], self._id2task[id(task2)]
        constraint = DifferentResources(idx1, idx2)
        self._constraints.different_resources.append(constraint)

        return constraint

    def add_consecutive(
        self, task1: Task, task2: Task, machine: Machine
    ) -> Consecutive:
        """
        Adds a constraint that the first task must be scheduled right before
        the second task on the given machine, meaning that no other task is
        allowed to be scheduled in-between.
        """
        idx1, idx2 = self._id2task[id(task1)], self._id2task[id(task2)]
        machine_idx = self._id2resource[id(machine)]
        constraint = Consecutive(idx1, idx2, machine_idx)
        self._constraints.consecutive.append(constraint)

        return constraint

    def add_if_then(
        self, pred: Task, succs: Union[Task, list[Task]]
    ) -> IfThen:
        """
        Adds a constraint that if the predecessor task is present, then at
        least one of the successor tasks must be present.
        """
        idx1 = self._id2task[id(pred)]
        succs = [succs] if isinstance(succs, Task) else succs
        idcs2 = [self._id2task[id(succ)] for succ in succs]
        constraint = IfThen(idx1, idcs2)
        self._constraints.if_then.append(constraint)

        return constraint

    def add_setup_time(
        self, machine: Machine, task1: Task, task2: Task, duration: int
    ) -> SetupTime:
        """
        Adds a setup time between two tasks on a machine.
        """
        machine_idx = self._id2resource[id(machine)]
        task_idx1 = self._id2task[id(task1)]
        task_idx2 = self._id2task[id(task2)]

        constraint = SetupTime(machine_idx, task_idx1, task_idx2, duration)
        self._constraints._setup_times.append(constraint)

        return constraint

    def set_objective(
        self,
        weight_makespan: int = 0,
        weight_tardy_jobs: int = 0,
        weight_total_tardiness: int = 0,
        weight_total_flow_time: int = 0,
        weight_total_earliness: int = 0,
        weight_max_tardiness: int = 0,
        weight_max_lateness: int = 0,
    ) -> Objective:
        """
        Sets the objective function in this model.
        """
        self._objective = Objective(
            weight_makespan=weight_makespan,
            weight_tardy_jobs=weight_tardy_jobs,
            weight_total_tardiness=weight_total_tardiness,
            weight_total_flow_time=weight_total_flow_time,
            weight_total_earliness=weight_total_earliness,
            weight_max_tardiness=weight_max_tardiness,
            weight_max_lateness=weight_max_lateness,
        )
        return self._objective

    def solve(
        self,
        solver: str = "ortools",
        time_limit: float = float("inf"),
        display: bool = True,
        num_workers: Optional[int] = None,
        initial_solution: Optional[Solution] = None,
        **kwargs,
    ) -> Result:
        """
        Solves the problem data instance created by the model.

        Parameters
        ----------
        solver
            The solver to use. Either ``'ortools'`` (default) or
            ``'cpoptimizer'``.
        time_limit
            The time limit for the solver in seconds. Default ``float('inf')``.
        display
            Whether to display the solver output. Default ``True``.
        num_workers
            The number of workers to use for parallel solving. If not
            specified, the default of the selected solver is used, which is
            typically the number of available CPU cores.
        initial_solution
            An initial solution to start the solver from. Default is no
            solution.
        kwargs
            Additional parameters passed to the solver.

        Returns
        -------
        Result
            A Result object containing the best found solution and additional
            information about the solver run.
        """
        return solve(
            self.data(),
            solver,
            time_limit,
            display,
            num_workers,
            initial_solution,
            **kwargs,
        )
