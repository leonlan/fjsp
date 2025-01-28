from itertools import product

import docplex.cp.modeler as cpo
import numpy as np
from docplex.cp.expression import interval_var
from docplex.cp.model import CpoModel

import pyjobshop.solvers.utils as utils
from pyjobshop.ProblemData import (
    Machine,
    NonRenewable,
    ProblemData,
    Renewable,
)

from .utils import presence_of
from .Variables import Variables


class Constraints:
    """
    Builds the core constraints of the CP Optimizer model.
    """

    def __init__(
        self, model: CpoModel, data: ProblemData, variables: Variables
    ):
        self._model = model
        self._data = data
        self._job_vars = variables.job_vars
        self._task_vars = variables.task_vars
        self._mode_vars = variables.mode_vars
        self._sequence_vars = variables.sequence_vars

    def _job_spans_tasks(self):
        """
        Ensures that the job variables span the related task variables.
        """
        model, data = self._model, self._data

        for idx, job in enumerate(data.jobs):
            job_var = self._job_vars[idx]
            task_vars = [self._task_vars[task] for task in job.tasks]

            if all(data.tasks[task].optional for task in job.tasks):
                # ``span`` requires at least one present interval variable
                # because the job interval is always present, so we add a
                # present dummy interval to be sure this is true.
                task_vars += [interval_var(name="dummy")]

            model.add(cpo.span(job_var, task_vars))

    def _select_one_mode(self):
        """
        Selects one mode for each task, ensuring that each task performs
        exactly one mode.
        """
        model, data = self._model, self._data
        task2modes = utils.task2modes(data)

        for task in range(data.num_tasks):
            mode_vars = [self._mode_vars[mode] for mode in task2modes[task]]
            model.add(cpo.alternative(self._task_vars[task], mode_vars))

    def _machines_no_overlap_and_setup_times(self):
        """
        Creates the no-overlap constraints for machines, ensuring that no two
        intervals in a sequence variable are overlapping. If setup times are
        available, the setup times are enforced as well.
        """
        model, data = self._model, self._data
        resource2modes = utils.resource2modes(data)

        for idx, resource in enumerate(data.resources):
            if not isinstance(resource, Machine):
                continue

            if not (modes := resource2modes[idx]):
                continue  # skip because cpo warns if there are no modes

            seq_var = self._sequence_vars[idx]
            setup_times = utils.setup_times_matrix(data)

            if setup_times is not None:
                # Slice the setup times matrix to get only durations for the
                # tasks corresponding to the modes of this machine.
                tasks = [data.modes[mode].task for mode in modes]
                matrix = setup_times[idx, :, :][np.ix_(tasks, tasks)]
                matrix = matrix if np.any(matrix > 0) else None
            else:
                matrix = None

            model.add(cpo.no_overlap(seq_var, matrix))

    def _renewable_capacity(self):
        """
        Creates capacity constraints for the renewable resources.
        """
        model, data = self._model, self._data
        res2modes, res2demands = utils.resource2modes_demands(data)

        for idx, resource in enumerate(data.resources):
            if not isinstance(resource, Renewable):
                continue

            pulses = [
                cpo.pulse(self._mode_vars[mode], demand)
                for (mode, demand) in zip(res2modes[idx], res2demands[idx])
                if demand > 0  # avoids cpo warnings
            ]
            model.add(model.sum(pulses) <= resource.capacity)

    def _non_renewable_capacity(self):
        """
        Creates capacity constraints for the non-renewable resources.
        """
        model, data = self._model, self._data
        res2modes, res2demands = utils.resource2modes_demands(data)

        for idx, resource in enumerate(data.resources):
            if not isinstance(resource, NonRenewable):
                continue

            usage = [
                presence_of(self._mode_vars[mode]) * demand
                for (mode, demand) in zip(res2modes[idx], res2demands[idx])
            ]
            model.add(model.sum(usage) <= resource.capacity)

    def _timing_constraints(self):
        """
        Creates constraints based on the task graph for task variables.
        """
        model, data = self._model, self._data

        for idx1, idx2, delay in data.constraints.start_before_start:
            task_var1 = self._task_vars[idx1]
            task_var2 = self._task_vars[idx2]
            model.add(cpo.start_before_start(task_var1, task_var2, delay))

        for idx1, idx2, delay in data.constraints.start_before_end:
            task_var1 = self._task_vars[idx1]
            task_var2 = self._task_vars[idx2]
            model.add(cpo.start_before_end(task_var1, task_var2, delay))

        for idx1, idx2, delay in data.constraints.end_before_start:
            task_var1 = self._task_vars[idx1]
            task_var2 = self._task_vars[idx2]
            model.add(cpo.end_before_start(task_var1, task_var2, delay))

        for idx1, idx2, delay in data.constraints.end_before_end:
            task_var1 = self._task_vars[idx1]
            task_var2 = self._task_vars[idx2]
            model.add(cpo.end_before_end(task_var1, task_var2, delay))

    def _identical_and_different_resource_constraints(self):
        """
        Creates constraints for identical and different resources constraints.
        """
        model, data = self._model, self._data

        for idx1, idx2 in data.constraints.identical_resources:
            for mode1, modes2 in utils.identical_modes(data, idx1, idx2):
                expr1 = presence_of(self._mode_vars[mode1])
                expr2 = sum(
                    presence_of(self._mode_vars[mode2]) for mode2 in modes2
                )
                model.add(expr1 <= expr2)

        for idx1, idx2 in data.constraints.different_resources:
            for mode1, modes2 in utils.different_modes(data, idx1, idx2):
                expr1 = presence_of(self._mode_vars[mode1])
                expr2 = sum(
                    presence_of(self._mode_vars[mode2]) for mode2 in modes2
                )
                model.add(expr1 <= expr2)

    def _if_then_constraints(self):
        """
        Creates constraints for the if-then constraints.
        """
        model, data = self._model, self._data

        for idx1, idcs2 in data.constraints.if_then:
            present1 = presence_of(self._task_vars[idx1])
            present2 = sum(presence_of(self._task_vars[idx]) for idx in idcs2)
            model.add(present1 <= present2)

    def _consecutive_constraints(self):
        """
        Creates the consecutive constraints.
        """
        model, data = self._model, self._data

        for idx1, idx2, machine in data.constraints.consecutive:
            seq_var = self._sequence_vars[machine]
            modes = utils.resource2modes(data)[machine]

            # Find the mode variables for these tasks on the machine.
            modes1 = [mode for mode in modes if data.modes[mode].task == idx1]
            modes2 = [mode for mode in modes if data.modes[mode].task == idx2]
            if not modes1 or not modes2:
                continue  # tasks not both assigned to the machine

            for mode1, mode2 in product(modes1, modes2):
                var1, var2 = self._mode_vars[mode1], self._mode_vars[mode2]
                model.add(cpo.previous(seq_var, var1, var2))

    def add_constraints(self):
        """
        Adds all the constraints to the CP model.
        """
        self._job_spans_tasks()
        self._select_one_mode()
        self._machines_no_overlap_and_setup_times()
        self._renewable_capacity()
        self._non_renewable_capacity()
        self._timing_constraints()
        self._identical_and_different_resource_constraints()
        self._if_then_constraints()
        self._consecutive_constraints()
