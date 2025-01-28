from dataclasses import dataclass, field

from ortools.sat.python.cp_model import (
    BoolVarT,
    CpModel,
    Domain,
    IntervalVar,
    IntVar,
)

import pyjobshop.solvers.utils as utils
from pyjobshop.constants import MAX_VALUE
from pyjobshop.ProblemData import Machine, ProblemData
from pyjobshop.Solution import Solution


@dataclass
class JobVar:
    """
    Variables that represent a job in the problem.

    Parameters
    ----------
    interval
        The interval variable representing the job.
    start
        The start time variable of the interval.
    duration
        The duration variable of the interval.
    end
        The end time variable of the interval.
    """

    interval: IntervalVar
    start: IntVar
    duration: IntVar
    end: IntVar


@dataclass
class TaskVar:
    """
    Variables that represent a task in the problem.

    Parameters
    ----------
    interval
        The interval variable representing the task.
    start
        The start time variable of the interval.
    duration
        The duration variable of the interval.
    end
        The end time variable of the interval.
    present
        The boolean variable indicating whether the interval is present.
    """

    interval: IntervalVar
    start: IntVar
    duration: IntVar
    end: IntVar
    present: BoolVarT


@dataclass
class ModeVar:
    """
    Variables that represent a possible processing mode of a task.

    Parameters
    ----------
    task_idx
        The index of the task.
    interval
        The optional interval variable representing the assignment of the task.
    start
        The start time variable of the interval.
    duration
        The duration variable of the interval.
    end
        The end time variable of the interval.
    present
        The boolean variable indicating whether the interval is present.
    """

    task_idx: int
    interval: IntervalVar
    start: IntVar
    duration: IntVar
    end: IntVar
    present: BoolVarT


@dataclass
class SequenceVar:
    """
    Represents a sequence of interval variables for all modes that use this
    machine.

    Parameters
    ----------
    mode_vars
        The mode interval variables belonging to this sequence.
    arcs
        The arc literals between each pair of intervals in the sequence
        indicating whether intervals are scheduled directly behind each other.
        Also includes arcs to and from a dummy node for each interval.
    is_active
        A boolean that indicates whether the sequence is active, meaning that a
        circuit constraint must be added for this machine. Default ``False``.

    Notes
    -----
    Sequence variables are lazily generated when activated by constraints that
    call the ``activate`` method. This avoids creating unnecessary variables
    when the sequence variable is not used in the model.

    """

    DUMMY = -1

    mode_vars: list[ModeVar]
    arcs: dict[tuple[int, int], BoolVarT] = field(default_factory=dict)
    is_active: bool = False

    def activate(self, m: CpModel):
        """
        Activates the sequence variable by creating all relevant literals.
        """
        if self.is_active:
            return

        self.is_active = True

        # The nodes in the graph are the indices of the mode variables,
        # plus the index of the dummy node.
        nodes = list(range(len(self.mode_vars))) + [self.DUMMY]

        self.arcs = {
            (i, j): m.new_bool_var(f"{i}->{j}") for i in nodes for j in nodes
        }


class Variables:
    """
    Manages the core variables of the OR-Tools model.
    """

    def __init__(self, model: CpModel, data: ProblemData):
        self._model = model
        self._data = data

        self._job_vars = self._make_job_variables()
        self._task_vars = self._make_task_variables()
        self._mode_vars = self._make_mode_variables()
        self._sequence_vars = self._make_sequence_variables()

    @property
    def job_vars(self) -> list[JobVar]:
        """
        Returns the job variables.
        """
        return self._job_vars

    @property
    def task_vars(self) -> list[TaskVar]:
        """
        Returns the task variables.
        """
        return self._task_vars

    @property
    def mode_vars(self) -> list[ModeVar]:
        """
        Returns the mode variables.
        """
        return self._mode_vars

    @property
    def sequence_vars(self) -> dict[int, SequenceVar]:
        """
        Returns the sequence variables.
        """
        return self._sequence_vars

    def _make_job_variables(self) -> list[JobVar]:
        """
        Creates an interval variable for each job.
        """
        model, data = self._model, self._data
        variables = []

        for idx, job in enumerate(data.jobs):
            name = f"J{idx}"
            start = model.new_int_var(
                lb=job.release_date,
                ub=MAX_VALUE,
                name=f"{name}_start",
            )
            duration = model.new_int_var(
                lb=0,
                ub=min(job.deadline - job.release_date, MAX_VALUE),
                name=f"{name}_duration",
            )
            end = model.new_int_var(
                lb=0,
                ub=min(job.deadline, MAX_VALUE),
                name=f"{name}_end",
            )
            interval = model.new_interval_var(
                start, duration, end, f"{name}_interval"
            )
            variables.append(JobVar(interval, start, duration, end))

        return variables

    def _make_task_variables(self) -> list[TaskVar]:
        """
        Creates an interval variable for each task.
        """
        model, data = self._model, self._data
        variables = []
        task_durations = utils.compute_task_durations(data)

        for idx, task in enumerate(data.tasks):
            name = f"T{idx}"
            start = model.new_int_var(
                lb=task.earliest_start,
                ub=min(task.latest_start, MAX_VALUE),
                name=f"{name}_start",
            )
            if task.fixed_duration:
                duration = model.new_int_var_from_domain(
                    Domain.from_values(task_durations[idx]), f"{name}_duration"
                )
            else:
                duration = model.new_int_var(
                    lb=min(task_durations[idx]),
                    ub=MAX_VALUE,
                    name=f"{name}_duration",
                )
            end = model.new_int_var(
                lb=task.earliest_end,
                ub=min(task.latest_end, MAX_VALUE),
                name=f"{name}_end",
            )
            present = (
                model.new_bool_var(f"{name}_present")
                if task.optional
                else model.new_constant(True)
            )
            interval = model.new_optional_interval_var(
                start, duration, end, present, f"interval_{task}"
            )
            variables.append(TaskVar(interval, start, duration, end, present))

        return variables

    def _make_mode_variables(self) -> list[ModeVar]:
        """
        Creates an optional interval variable for mode.
        """
        model, data = self._model, self._data
        variables = []

        for idx, mode in enumerate(data.modes):
            task = data.tasks[mode.task]
            name = f"M{idx}_{mode.task}"
            start = model.new_int_var(
                lb=task.earliest_start,
                ub=min(task.latest_start, MAX_VALUE),
                name=f"{name}_start",
            )
            duration = model.new_int_var(
                lb=mode.duration,
                ub=mode.duration if task.fixed_duration else MAX_VALUE,
                name=f"{name}_duration",
            )
            end = model.new_int_var(
                lb=task.earliest_end,
                ub=min(task.latest_end, MAX_VALUE),
                name=f"{name}_start",
            )
            present = model.new_bool_var(f"{name}_present")
            interval = model.new_optional_interval_var(
                start, duration, end, present, f"{name}_interval"
            )
            var = ModeVar(
                task_idx=mode.task,
                interval=interval,
                start=start,
                duration=duration,
                end=end,
                present=present,
            )
            variables.append(var)

        return variables

    def _make_sequence_variables(self) -> dict[int, SequenceVar]:
        """
        Creates a sequence variable for each machine.
        """
        data = self._data
        resource2modes = utils.resource2modes(data)
        variables: dict[int, SequenceVar] = {}

        for idx, resource in enumerate(data.resources):
            if isinstance(resource, Machine):
                modes = resource2modes[idx]
                intervals = [self.mode_vars[mode] for mode in modes]
                variables[idx] = SequenceVar(intervals)

        return variables

    def warmstart(self, solution: Solution):
        """
        Warmstarts the variables based on the given solution.
        """
        model, data = self._model, self._data
        job_vars, task_vars, mode_vars = (
            self.job_vars,
            self.task_vars,
            self.mode_vars,
        )

        model.clear_hints()

        for idx in range(data.num_jobs):
            job = data.jobs[idx]
            job_var = job_vars[idx]
            sol_tasks = [solution.tasks[task] for task in job.tasks]

            job_start = min(task.start for task in sol_tasks)
            job_end = max(task.end for task in sol_tasks)

            model.add_hint(job_var.start, job_start)
            model.add_hint(job_var.duration, job_end - job_start)
            model.add_hint(job_var.end, job_end)

        for idx in range(data.num_tasks):
            task_var = task_vars[idx]
            sol_task = solution.tasks[idx]

            model.add_hint(task_var.start, sol_task.start)
            model.add_hint(task_var.duration, sol_task.end - sol_task.start)
            model.add_hint(task_var.end, sol_task.end)

            if data.tasks[idx].optional:
                # OR-Tools complains about adding presence hints to interval
                # variables that are always present (i.e., non-optional tasks).
                model.add_hint(task_var.present, sol_task.present)

        for idx in range(len(data.modes)):
            var = mode_vars[idx]
            mode = data.modes[idx]
            sol_task = solution.tasks[mode.task]

            model.add_hint(var.start, sol_task.start)
            model.add_hint(var.duration, sol_task.end - sol_task.start)
            model.add_hint(var.end, sol_task.end)
            model.add_hint(var.present, idx == sol_task.mode)
