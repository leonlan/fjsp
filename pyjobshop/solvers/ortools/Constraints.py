import numpy as np
from ortools.sat.python.cp_model import BoolVarT, CpModel, LinearExpr

import pyjobshop.solvers.utils as utils
from pyjobshop.constants import MAX_VALUE
from pyjobshop.ProblemData import (
    Machine,
    NonRenewable,
    ProblemData,
    Renewable,
)
from pyjobshop.solvers.ortools.Variables import Variables


class Constraints:
    """
    Builds the core constraints of the OR-Tools model.
    """

    def __init__(
        self, model: CpModel, data: ProblemData, variables: Variables
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
            task_starts = []
            task_ends = []

            for task in job.tasks:
                task_var = self._task_vars[task]

                if data.tasks[task].optional:
                    # When tasks are absent, they should not restrict the job's
                    # start and end times.
                    task_start = model.new_int_var(0, MAX_VALUE, "")
                    task_end = model.new_int_var(0, MAX_VALUE, "")

                    expr = task_start == task_var.start
                    model.add(expr).only_enforce_if(task_var.present)

                    expr = task_end == task_var.end
                    model.add(expr).only_enforce_if(task_var.present)
                else:
                    task_start = task_var.start
                    task_end = task_var.end

                task_starts.append(task_start)
                task_ends.append(task_end)

            model.add_min_equality(job_var.start, task_starts)
            model.add_max_equality(job_var.end, task_ends)

    def _select_one_mode(self):
        """
        Selects one mode for each task, ensuring that each task performs
        exactly one mode.
        """
        model, data = self._model, self._data
        task2modes = utils.task2modes(data)

        for task in range(data.num_tasks):
            task_var = self._task_vars[task]

            # Select exactly one optional interval variable for each task.
            presences = [
                self._mode_vars[mode].present for mode in task2modes[task]
            ]
            model.add(sum(presences) == task_var.present)

            for mode in task2modes[task]:
                mode_var = self._mode_vars[mode]
                both_present = [task_var.present, mode_var.present]

                sync_start = task_var.start == mode_var.start
                model.add(sync_start).only_enforce_if(both_present)

                sync_duration = task_var.duration == mode_var.duration
                model.add(sync_duration).only_enforce_if(both_present)

                sync_end = task_var.end == mode_var.end
                model.add(sync_end).only_enforce_if(both_present)

    def _machines_no_overlap(self):
        """
        Creates no-overlap constraints for machines.
        """
        model, data = self._model, self._data

        for idx, resource in enumerate(data.resources):
            if not isinstance(resource, Machine):
                continue

            seq_var = self._sequence_vars[idx]
            mode_vars = [var.interval for var in seq_var.mode_vars]
            model.add_no_overlap(mode_vars)

    def _renewable_capacity(self):
        """
        Creates capacity constraints for the renewable resources.
        """
        model, data = self._model, self._data
        mode_vars = self._mode_vars
        res2modes, res2demands = utils.resource2modes_demands(data)

        for idx, resource in enumerate(data.resources):
            if not isinstance(resource, Renewable):
                continue

            intervals = [mode_vars[mode].interval for mode in res2modes[idx]]
            demands = res2demands[idx]
            model.add_cumulative(intervals, demands, resource.capacity)

    def _non_renewable_capacity(self):
        """
        Creates capacity constraints for the non-renewable resources.
        """
        model, data = self._model, self._data
        mode_vars = self._mode_vars
        res2modes, res2demands = utils.resource2modes_demands(data)

        for idx, resource in enumerate(data.resources):
            if not isinstance(resource, NonRenewable):
                continue

            precenses = [mode_vars[mode].present for mode in res2modes[idx]]
            demands = res2demands[idx]
            usage = LinearExpr.weighted_sum(precenses, demands)
            model.add(usage <= resource.capacity)

    def _timing_constraints(self):
        """
        Creates constraints based on the timing relationship between tasks.
        """
        model, data = self._model, self._data

        for idx1, idx2 in data.constraints.start_at_start:
            task_var1 = self._task_vars[idx1]
            task_var2 = self._task_vars[idx2]
            both_present = [task_var1.present, task_var2.present]

            expr = task_var1.start == task_var2.start
            model.add(expr).only_enforce_if(both_present)

        for idx1, idx2 in data.constraints.start_at_end:
            task_var1 = self._task_vars[idx1]
            task_var2 = self._task_vars[idx2]
            both_present = [task_var1.present, task_var2.present]
            expr = task_var1.start == task_var2.end
            model.add(expr).only_enforce_if(both_present)

        for idx1, idx2 in data.constraints.start_before_start:
            task_var1 = self._task_vars[idx1]
            task_var2 = self._task_vars[idx2]
            both_present = [task_var1.present, task_var2.present]
            expr = task_var1.start <= task_var2.start
            model.add(expr).only_enforce_if(both_present)

        for idx1, idx2 in data.constraints.start_before_end:
            task_var1 = self._task_vars[idx1]
            task_var2 = self._task_vars[idx2]
            both_present = [task_var1.present, task_var2.present]
            expr = task_var1.start <= task_var2.end
            model.add(expr).only_enforce_if(both_present)

        for idx1, idx2 in data.constraints.end_at_start:
            task_var1 = self._task_vars[idx1]
            task_var2 = self._task_vars[idx2]
            both_present = [task_var1.present, task_var2.present]
            expr = task_var1.end == task_var2.start
            model.add(expr).only_enforce_if(both_present)

        for idx1, idx2 in data.constraints.end_at_end:
            task_var1 = self._task_vars[idx1]
            task_var2 = self._task_vars[idx2]
            both_present = [task_var1.present, task_var2.present]
            expr = task_var1.end == task_var2.end
            model.add(expr).only_enforce_if(both_present)

        for idx1, idx2 in data.constraints.end_before_start:
            task_var1 = self._task_vars[idx1]
            task_var2 = self._task_vars[idx2]
            both_present = [task_var1.present, task_var2.present]
            expr = task_var1.end <= task_var2.start
            model.add(expr).only_enforce_if(both_present)

        for idx1, idx2 in data.constraints.end_before_end:
            task_var1 = self._task_vars[idx1]
            task_var2 = self._task_vars[idx2]
            both_present = [task_var1.present, task_var2.present]
            expr = task_var1.end <= task_var2.end
            model.add(expr).only_enforce_if(both_present)

    def _identical_and_different_resource_constraints(self):
        """
        Creates constraints for the same and different resource constraints.
        """
        model, data = self._model, self._data
        task2modes = utils.task2modes(data)

        for idx1, idx2 in data.constraints.identical_resources:
            identical = utils.find_modes_with_identical_resources(
                data, idx1, idx2
            )

            modes1 = task2modes[idx1]
            for mode1 in modes1:
                identical_modes2 = identical[mode1]
                var1 = self._mode_vars[mode1].is_present
                vars2 = [
                    self._mode_vars[mode2].is_present
                    for mode2 in identical_modes2
                ]
                model.add(sum(vars2) >= var1)

        for idx1, idx2 in data.constraints.different_resources:
            disjoint = utils.find_modes_with_disjoint_resources(
                data, idx1, idx2
            )
            modes1 = task2modes[idx1]
            for mode1 in modes1:
                disjoint_modes2 = disjoint[mode1]
                var1 = self._mode_vars[mode1].is_present
                vars2 = [
                    self._mode_vars[mode2].is_present
                    for mode2 in disjoint_modes2
                ]
                model.add(sum(vars2) >= var1)

    def _if_then_constraints(self):
        """
        Creates the if-then constraints.
        """
        model, data = self._model, self._data

        for (idx1, idcs2), constraints in data.constraints.items():
            if Constraint.IF_THEN not in constraints:
                continue

            pred = self._task_vars[idx1].present
            succs = sum(self._task_vars[idx2].present for idx2 in idcs2)
            model.add(pred <= succs)

    def _activate_setup_times(self):
        """
        Activates the sequence variables for resources that have setup times.
        The ``_circuit_constraints`` function will in turn add constraints to
        the CP-SAT model to enforce setup times.
        """
        model, data = self._model, self._data
        setup_times = utils.setup_times_matrix(data)

        for idx, resource in enumerate(data.resources):
            if not isinstance(resource, Machine):
                continue

            if setup_times is not None and np.any(setup_times[idx]):
                self._sequence_vars[idx].activate(model)

    def _consecutive_constraints(self):
        """
        Creates the consecutive constraints.
        """
        model, data = self._model, self._data

        for idx1, idx2 in data.constraints.consecutive:
            intersecting = utils.find_modes_with_intersecting_resources(
                data, idx1, idx2
            )
            for mode1, mode2, resources in intersecting:
                for resource in resources:
                    if not isinstance(data.resources[resource], Machine):
                        continue

                    seq_var = self._sequence_vars[resource]
                    seq_var.activate(model)
                    var1 = self._mode_vars[mode1]
                    var2 = self._mode_vars[mode2]

                    idx1 = seq_var.mode_vars.index(var1)
                    idx2 = seq_var.mode_vars.index(var2)
                    arc = seq_var.arcs[idx1, idx2]
                    both_present = [var1.present, var2.present]

                    model.add(arc == 1).only_enforce_if(both_present)

    def _circuit_constraints(self):
        """
        Creates the circuit constraints for each machine, if activated by
        sequencing constraints (consecutive and setup times).
        """
        model, data = self._model, self._data
        setup_times = utils.setup_times_matrix(data)

        for idx, resource in enumerate(data.resources):
            if not isinstance(resource, Machine):
                continue

            seq_var = self._sequence_vars[idx]
            if not seq_var.is_active:
                # No sequencing constraints active. Skip the creation of
                # (expensive) circuit constraints.
                continue

            modes = seq_var.mode_vars
            arcs = seq_var.arcs

            # Add dummy node self-arc to allow empty circuits.
            empty = model.new_bool_var("")
            graph: list[tuple[int, int, BoolVarT]] = [(-1, -1, empty)]

            for idx1, var1 in enumerate(modes):
                # Arcs from and to the dummy node.
                graph.append((-1, idx1, model.new_bool_var("")))
                graph.append((idx1, -1, model.new_bool_var("")))

                # Self arc if the task is not present.
                graph.append((idx1, idx1, ~var1.present))

                # If the circuit is empty then the var should not be present.
                model.add_implication(empty, ~var1.present)

                for idx2, var2 in enumerate(modes):
                    if idx1 == idx2:
                        continue

                    arc = arcs[idx1, idx2]
                    graph.append((idx1, idx2, arc))

                    model.add_implication(arc, var1.present)
                    model.add_implication(arc, var2.present)

                    setup = (
                        setup_times[idx, var1.task_idx, var2.task_idx]
                        if setup_times is not None
                        else 0
                    )
                    expr = var1.end + setup <= var2.start
                    model.add(expr).only_enforce_if(arc)

            model.add_circuit(graph)

    def add_constraints(self):
        """
        Adds all the constraints to the CP model.
        """
        self._job_spans_tasks()
        self._select_one_mode()
        self._machines_no_overlap()
        self._renewable_capacity()
        self._non_renewable_capacity()
        self._timing_constraints()
        self._identical_and_different_resource_constraints()
        self._if_then_constraints()
        self._activate_setup_times()
        self._consecutive_constraints()

        # From here onwards we know which sequence constraints are active.
        self._circuit_constraints()
