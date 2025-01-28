from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, TypeVar, Union

from pyjobshop.constants import MAX_VALUE

_T = TypeVar("_T")


class Job:
    """
    Simple dataclass for storing job related data.

    Parameters
    ----------
    weight
        The weight of the job, used as multiplicative factor in the
        objective function. Default ``1``.
    release_date
        The earliest time that the job may start. Default ``0``.
    deadline
        The latest time by which the job must be completed. Note that a
        deadline is different from a due date; the latter does not restrict
        the latest completion time.
        Default :const:`~pyjobshop.constants.MAX_VALUE`.
    due_date
        The latest time that the job should be completed before incurring
        penalties. Default ``None``, meaning that there is no due date.
    tasks
        List of task indices that belong to this job. Default ``None``,
        which initializes an empty list.
    name
        Name of the job.
    """

    def __init__(
        self,
        weight: int = 1,
        release_date: int = 0,
        deadline: int = MAX_VALUE,
        due_date: Optional[int] = None,
        tasks: Optional[list[int]] = None,
        name: str = "",
    ):
        if weight < 0:
            raise ValueError("Weight must be non-negative.")

        if release_date < 0:
            raise ValueError("Release date must be non-negative.")

        if deadline < 0:
            raise ValueError("Deadline must be non-negative.")

        if release_date > deadline:
            raise ValueError("Must have release_date <= deadline.")

        if due_date is not None and due_date < 0:
            raise ValueError("Due date must be non-negative.")

        self._weight = weight
        self._release_date = release_date
        self._deadline = deadline
        self._due_date = due_date
        self._tasks = [] if tasks is None else tasks
        self._name = name

    @property
    def weight(self) -> int:
        """
        The weight of the job, used as multiplicative factor in the objective
        function.
        """
        return self._weight

    @property
    def release_date(self) -> int:
        """
        The earliest time that the job may start.
        """
        return self._release_date

    @property
    def deadline(self) -> int:
        """
        The latest time by which the job must be completed.
        """
        return self._deadline

    @property
    def due_date(self) -> Optional[int]:
        """
        The latest time that the job should be completed before incurring
        penalties.
        """
        return self._due_date

    @property
    def tasks(self) -> list[int]:
        """
        List of task indices that belong to this job.
        """
        return self._tasks

    @property
    def name(self) -> str:
        """
        Name of the job.
        """
        return self._name

    def add_task(self, idx: int):
        """
        Adds a task index to the job.

        Parameters
        ----------
        idx
            Task index to add.
        """
        self._tasks.append(idx)


class Machine:
    """
    A machine resource is a specialized resource that only processes one task
    at a time and can handle sequencing constraints.

    Parameters
    ----------
    name
        Name of the machine.
    """

    def __init__(self, name: str = ""):
        self._name = name

    @property
    def name(self) -> str:
        """
        Name of the machine.
        """
        return self._name


class Renewable:
    """
    A renewable resource that replenishes its capacity after each task
    completion.

    Parameters
    ----------
    capacity
        Capacity of the resource.
    name
        Name of the resource.
    """

    def __init__(self, capacity: int, name: str = ""):
        if capacity < 0:
            raise ValueError("Capacity must be non-negative.")

        self._capacity = capacity
        self._name = name

    @property
    def capacity(self) -> int:
        """
        Capacity of the resource.
        """
        return self._capacity

    @property
    def name(self) -> str:
        """
        Name of the resource.
        """
        return self._name


class NonRenewable:
    """
    A non-renewable resource that does not replenish its capacity.

    Parameters
    ----------
    capacity
        Capacity of the resource.
    name
        Name of the resource.
    """

    def __init__(self, capacity: int, name: str = ""):
        if capacity < 0:
            raise ValueError("Capacity must be non-negative.")

        self._capacity = capacity
        self._name = name

    @property
    def capacity(self) -> int:
        """
        Capacity of the resource.
        """
        return self._capacity

    @property
    def name(self) -> str:
        """
        Name of the resource.
        """
        return self._name


Resource = Union[Machine, Renewable, NonRenewable]


class Task:
    """
    Simple dataclass for storing task related data.

    Parameters
    ----------
    job
        The index of the job that this task belongs to. None if the task
        does not belong to any job. Default ``None``.
    earliest_start
        Earliest start time of the task. Default ``0``.
    latest_start
        Latest start time of the task.
        Default :const:`~pyjobshop.constants.MAX_VALUE`.
    earliest_end
        Earliest end time of the task. Default ``0``.
    latest_end
        Latest end time of the task.
        Default :const:`~pyjobshop.constants.MAX_VALUE`.
    fixed_duration
        Whether the task has a fixed duration. A fixed duration means that
        the task duration is precisely the processing time (on a given
        resource). If the duration is not fixed, then the task duration
        can take longer than the processing time, e.g., due to blocking.
        Default ``True``.
    optional
        Whether the task is optional. Default ``False``.
    name
        Name of the task.
    """

    def __init__(
        self,
        job: Optional[int] = None,
        earliest_start: int = 0,
        latest_start: int = MAX_VALUE,
        earliest_end: int = 0,
        latest_end: int = MAX_VALUE,
        fixed_duration: bool = True,
        optional: bool = False,
        name: str = "",
    ):
        if earliest_start > latest_start:
            raise ValueError("earliest_start must be <= latest_start.")

        if earliest_end > latest_end:
            raise ValueError("earliest_end must be <= latest_end.")

        self._job = job
        self._earliest_start = earliest_start
        self._latest_start = latest_start
        self._earliest_end = earliest_end
        self._latest_end = latest_end
        self._fixed_duration = fixed_duration
        self._optional = optional
        self._name = name

    @property
    def job(self) -> Optional[int]:
        """
        The index of the job that this task belongs to. None if the task
        does not belong to any job.
        """
        return self._job

    @property
    def earliest_start(self) -> int:
        """
        Earliest start time of the task.
        """
        return self._earliest_start

    @property
    def latest_start(self) -> int:
        """
        Latest start time of the task.
        """
        return self._latest_start

    @property
    def earliest_end(self) -> int:
        """
        Earliest end time of the task.
        """
        return self._earliest_end

    @property
    def latest_end(self) -> int:
        """
        Latest end time of the task.
        """
        return self._latest_end

    @property
    def fixed_duration(self) -> bool:
        """
        Whether the task has a fixed duration.
        """
        return self._fixed_duration

    @property
    def optional(self) -> bool:
        """
        Whether the task is optional.
        """
        return self._optional

    @property
    def name(self) -> str:
        """
        Name of the task.
        """
        return self._name


class Mode:
    """
    Simple dataclass for storing processing mode data.

    Parameters
    ----------
    task
        Task index that this mode belongs to.
    resources
        List of resources that are required for this mode.
    duration
        Processing duration of this mode.
    demands
        Optional list of demands for each resource for this mode. If ``None``
        is given, then the demands are initialized as list of zeros with the
        same length as the resources.
    """

    def __init__(
        self,
        task: int,
        resources: list[int],
        duration: int,
        demands: Optional[list[int]] = None,
    ):
        if len(set(resources)) != len(resources):
            raise ValueError("Mode resources must be unique.")

        if duration < 0:
            raise ValueError("Mode duration must be non-negative.")

        demands = demands if demands is not None else [0] * len(resources)
        if any(demand < 0 for demand in demands):
            raise ValueError("Mode demands must be non-negative.")

        if len(resources) != len(demands):
            raise ValueError("resources and demands must have same length.")

        self._task = task
        self._resources = resources
        self._duration = duration
        self._demands = demands

    @property
    def task(self) -> int:
        return self._task

    @property
    def resources(self) -> list[int]:
        return self._resources

    @property
    def duration(self) -> int:
        return self._duration

    @property
    def demands(self) -> list[int]:
        return self._demands

    def __eq__(self, other) -> bool:
        return (
            self.task == other.task
            and self.resources == other.resources
            and self.duration == other.duration
            and self.demands == other.demands
        )


class StartBeforeStart(NamedTuple):
    """
    Start task 1 (:math:`s_1`) before task 2 starts (:math:`s_2`), with an
    optional delay :math:`d`. That is,

    .. math::
        s_1 + d \\leq s_2.
    """

    task1: int
    task2: int
    delay: int = 0


class StartBeforeEnd(NamedTuple):
    """
    Start task 1 (:math:`s_1`) before task 2 ends (:math:`e_2`), with an
    optional delay :math:`d`. That is,

    .. math::
        s_1 + d \\leq e_2.
    """

    task1: int
    task2: int
    delay: int = 0


class EndBeforeStart(NamedTuple):
    """
    End task 1 (:math:`e_1`) before task 2 starts (:math:`s_2`), with an
    optional delay :math:`d`. That is,

    .. math::
        e_1 + d \\leq s_2.
    """

    task1: int
    task2: int
    delay: int = 0


class EndBeforeEnd(NamedTuple):
    """
    End task 1 (:math:`e_1`) before task 2 ends (:math:`e_2`), with an
    optional delay :math:`d`. That is,

    .. math::
        e_1 + d \\leq e_2.
    """

    task1: int
    task2: int
    delay: int = 0


class IdenticalResources(NamedTuple):
    """
    Select modes for task 1 and task 2 that use the same resources.

    Let :math:`m_1, m_2` be the selected modes of task 1 and task 2, and let
    :math:`R_m` denote the resources required by mode :math:`m`. This
    constraint ensures that

    .. math::
        R_{m_1} = R_{m_2}.
    """

    task1: int
    task2: int


class DifferentResources(NamedTuple):
    """
    Select modes for task 1 and task 2 that use different resources.

    Let :math:`m_1, m_2` be the selected modes of task 1 and task 2, and let
    :math:`R_m` denote the resources required by mode :math:`m`. This
    constraint ensures that

    .. math::
        R_{m_1} \\cap R_{m_2} = \\emptyset.
    """

    task1: int
    task2: int


class IfThen(NamedTuple):
    """
    If predecessor task is present, then at least one of the successor tasks
    must be present.
    """

    predecessor: int
    successors: list[int]


class Consecutive(NamedTuple):
    """
    Sequence task 1 and task 2 consecutively on the given machine, meaning that
    no other task is allowed to be scheduled in-between.
    """

    task1: int
    task2: int
    machine: int


class SetupTime(NamedTuple):
    """
    Sequence-dependent setup time between task 1 and task 2 on the given
    machine.

    Let :math:`e_1` be the end time of task 1 and let :math:`s_2` be the
    start time of task 2. If the selected modes of task 1 and task 2 both
    require the given machine, then this constraint ensures that

    .. math::
        e_1 + d \\leq s_2,

    where :math:`d` is the setup time duration. Note that this also implies
    an end-before-start relationship between task 1 and task 2.
    """

    machine: int
    task1: int
    task2: int
    duration: int


class Constraints:
    """
    Container class for storing all constraints.
    """

    def __init__(
        self,
        start_before_start: Optional[list[StartBeforeStart]] = None,
        start_before_end: Optional[list[StartBeforeEnd]] = None,
        end_before_start: Optional[list[EndBeforeStart]] = None,
        end_before_end: Optional[list[EndBeforeEnd]] = None,
        identical_resources: Optional[list[IdenticalResources]] = None,
        different_resources: Optional[list[DifferentResources]] = None,
        if_then: Optional[list[IfThen]] = None,
        consecutive: Optional[list[Consecutive]] = None,
        setup_times: Optional[list[SetupTime]] = None,
    ):
        self._start_before_start = start_before_start or []
        self._start_before_end = start_before_end or []
        self._end_before_start = end_before_start or []
        self._end_before_end = end_before_end or []
        self._identical_resources = identical_resources or []
        self._different_resources = different_resources or []
        self._if_then = if_then or []
        self._consecutive = consecutive or []
        self._setup_times = setup_times or []

    def __eq__(self, other) -> bool:
        return (
            self.start_before_start == other.start_before_start
            and self.start_before_end == other.start_before_end
            and self.end_before_start == other.end_before_start
            and self.end_before_end == other.end_before_end
            and self.identical_resources == other.identical_resources
            and self.different_resources == other.different_resources
            and self.if_then == other.if_then
            and self.consecutive == other.consecutive
            and self.setup_times == other.setup_times
        )

    def __len__(self) -> int:
        return (
            len(self.start_before_start)
            + len(self.start_before_end)
            + len(self.end_before_start)
            + len(self.end_before_end)
            + len(self.identical_resources)
            + len(self.different_resources)
            + len(self.if_then)
            + len(self.consecutive)
            + len(self._setup_times)
        )

    @property
    def start_before_start(self) -> list[StartBeforeStart]:
        """
        Returns the list of start-before-start constraints.
        """
        return self._start_before_start

    @property
    def start_before_end(self) -> list[StartBeforeEnd]:
        """
        Returns the list of start-before-end constraints.
        """
        return self._start_before_end

    @property
    def end_before_start(self) -> list[EndBeforeStart]:
        """
        Returns the list of end-before-start constraints.
        """
        return self._end_before_start

    @property
    def end_before_end(self) -> list[EndBeforeEnd]:
        """
        Returns the list of end-before-end constraints.
        """
        return self._end_before_end

    @property
    def identical_resources(self) -> list[IdenticalResources]:
        """
        Returns the list of identical resources constraints.
        """
        return self._identical_resources

    @property
    def different_resources(self) -> list[DifferentResources]:
        """
        Returns the list of different resources constraints.
        """
        return self._different_resources

    @property
    def if_then(self) -> list[IfThen]:
        """
        Returns the list of if-then constraints.
        """
        return self._if_then

    @property
    def consecutive(self) -> list[Consecutive]:
        """
        Returns the list of consecutive task constraints.
        """
        return self._consecutive

    @property
    def setup_times(self) -> list[SetupTime]:
        """
        Returns the list of setup times constraints.
        """
        return self._setup_times


@dataclass
class Objective:
    """
    Represents a weighted sum of the following objective functions:

    * Makespan
    * Number of tardy jobs
    * Total flow time
    * Total tardiness
    * Total earliness
    * Maximum tardiness
    * Maximum lateness

    .. note::
        Use :attr:`Job.weight` to set a specific job's weight in the
        objective function.
    """

    weight_makespan: int = 0
    weight_tardy_jobs: int = 0
    weight_total_flow_time: int = 0
    weight_total_tardiness: int = 0
    weight_total_earliness: int = 0
    weight_max_tardiness: int = 0
    weight_max_lateness: int = 0


class ProblemData:
    """
    Class that contains all data needed to solve the scheduling problem.

    Parameters
    ----------
    jobs
        List of jobs.
    resources
        List of resources.
    tasks
        List of tasks.
    modes
        List of processing modes of tasks.
    constraints
        The constraints of this problem data instance. Default is no
        constraints.
    objective
        The objective function. Default is minimizing the makespan.
    """

    def __init__(
        self,
        jobs: list[Job],
        resources: Sequence[Resource],
        tasks: list[Task],
        modes: list[Mode],
        constraints: Optional[Constraints] = None,
        objective: Optional[Objective] = None,
    ):
        self._jobs = jobs
        self._resources = resources
        self._tasks = tasks
        self._modes = modes
        self._constraints = (
            constraints if constraints is not None else Constraints()
        )
        self._objective = (
            objective
            if objective is not None
            else Objective(weight_makespan=1)
        )

        self._validate_parameters()

    def _validate_parameters(self):
        """
        Validates the problem data parameters.
        """
        num_res = self.num_resources
        num_tasks = self.num_tasks

        for idx, job in enumerate(self.jobs):
            if any(task < 0 or task >= num_tasks for task in job.tasks):
                msg = f"Job {idx} references to unknown task index."
                raise ValueError(msg)

        for idx, task in enumerate(self.tasks):
            if task.job is not None:
                if task.job < 0 or task.job >= len(self.jobs):
                    msg = f"Task {idx} references to unknown job index."
                    raise ValueError(msg)

        for idx, mode in enumerate(self.modes):
            if mode.task < 0 or mode.task >= num_tasks:
                raise ValueError(f"Mode {idx} references unknown task index.")

            for resource in mode.resources:
                if resource < 0 or resource >= num_res:
                    msg = f"Mode {idx} references unknown resource index."
                    raise ValueError(msg)

        missing = set(range(num_tasks)) - {mode.task for mode in self.modes}
        if missing := sorted(missing):
            raise ValueError(f"Processing modes missing for tasks {missing}.")

        infeasible_modes = Counter()
        num_modes = Counter()

        for mode in self.modes:
            num_modes[mode.task] += 1
            infeasible_modes[mode.task] += any(
                # Assumes that machines have zero capacity.
                demand > getattr(self.resources[res], "capacity", 0)
                for demand, res in zip(mode.demands, mode.resources)
            )

        for task, count in num_modes.items():
            if infeasible_modes[task] == count:
                msg = f"All modes for task {task} have infeasible demands."
                raise ValueError(msg)

        for res_idx, *_, duration in self.constraints.setup_times:
            if duration < 0:
                raise ValueError("Setup time must be non-negative.")

            is_machine = isinstance(self.resources[res_idx], Machine)
            has_setup_times = duration > 0

            if not is_machine and has_setup_times:
                raise ValueError("Setup times only allowed for machines.")

        if (
            self.objective.weight_tardy_jobs > 0
            or self.objective.weight_total_tardiness > 0
            or self.objective.weight_total_earliness > 0
            or self.objective.weight_max_tardiness > 0
            or self.objective.weight_max_lateness > 0
        ):
            if any(job.due_date is None for job in self.jobs):
                msg = "Job due dates required for due date-based objectives."
                raise ValueError(msg)

    def replace(
        self,
        jobs: Optional[list[Job]] = None,
        resources: Optional[Sequence[Resource]] = None,
        tasks: Optional[list[Task]] = None,
        modes: Optional[list[Mode]] = None,
        constraints: Optional[Constraints] = None,
        objective: Optional[Objective] = None,
    ) -> "ProblemData":
        """
        Returns a new ProblemData instance with possibly replaced data. If a
        parameter is not provided, the original data is deepcopied instead.

        Parameters
        ----------
        jobs
            Optional list of jobs.
        resources
            Optional list of resources.
        tasks
            Optional list of tasks.
        modes
            Optional processing modes of tasks.
        constraints
            Optional constraints.
        objective
            Optional objective function.

        Returns
        -------
        ProblemData
            A new ProblemData instance with possibly replaced data.
        """

        def _deepcopy_if_none(value: Optional[_T], default: _T) -> _T:
            return value if value is not None else deepcopy(default)

        jobs = _deepcopy_if_none(jobs, self.jobs)
        resources = _deepcopy_if_none(resources, self.resources)
        tasks = _deepcopy_if_none(tasks, self.tasks)
        modes = _deepcopy_if_none(modes, self.modes)
        constraints = _deepcopy_if_none(constraints, self.constraints)
        objective = _deepcopy_if_none(objective, self.objective)

        return ProblemData(
            jobs=jobs,
            resources=resources,
            tasks=tasks,
            modes=modes,
            constraints=constraints,
            objective=objective,
        )

    @property
    def jobs(self) -> list[Job]:
        """
        Returns the job data of this problem instance.
        """
        return self._jobs

    @property
    def resources(self) -> Sequence[Resource]:
        """
        Returns the resource data of this problem instance.
        """
        return self._resources

    @property
    def tasks(self) -> list[Task]:
        """
        Returns the task data of this problem instance.
        """
        return self._tasks

    @property
    def modes(self) -> list[Mode]:
        """
        Returns the processing modes of this problem instance.
        """
        return self._modes

    @property
    def constraints(self) -> Constraints:
        """
        Returns the constraints of this problem instance.
        """
        return self._constraints

    @property
    def objective(self) -> Objective:
        """
        Returns the objective function of this problem instance.
        """
        return self._objective

    @property
    def num_jobs(self) -> int:
        """
        Returns the number of jobs in this instance.
        """
        return len(self._jobs)

    @property
    def num_resources(self) -> int:
        """
        Returns the number of resources in this instance.
        """
        return len(self._resources)

    @property
    def num_tasks(self) -> int:
        """
        Returns the number of tasks in this instance.
        """
        return len(self._tasks)

    @property
    def num_modes(self) -> int:
        """
        Returns the number of modes in this instance.
        """
        return len(self._modes)

    @property
    def num_constraints(self) -> int:
        """
        Returns the number of constraints in this instance.
        """
        return len(self._constraints)
