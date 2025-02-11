from enum import Enum
from pathlib import Path
from typing import Union

import fjsplib
import psplib

from .Model import Model
from .ProblemData import ProblemData


class InstanceFormat(str, Enum):
    """
    Supported instance formats.
    """

    FJSPLIB = "fjsplib"
    PSPLIB = "psplib"
    MPLIB = "mplib"
    PATTERSON = "patterson"
    RCPSP_MAX = "rcpsp_max"


def read(
    loc: Union[str, Path],
    instance_format: InstanceFormat = InstanceFormat.FJSPLIB,
) -> ProblemData:
    """
    Reads an FJSPLIB instance and returns a ProblemData object.
    """
    if instance_format == InstanceFormat.FJSPLIB:
        return _read_fjslib(loc)
    elif instance_format in [
        InstanceFormat.PSPLIB,
        InstanceFormat.MPLIB,
        InstanceFormat.PATTERSON,
        InstanceFormat.RCPSP_MAX,
    ]:
        instance = psplib.parse(loc, instance_format)
        return _project_instance_to_data(instance)

    raise ValueError(f"Unknown instance format: {instance_format}")


def _read_fjslib(loc: Union[str, Path]) -> ProblemData:
    """
    Reads an FJSPLIB instance and returns a ProblemData object.
    """
    instance = fjsplib.read(loc)

    model = Model()

    jobs = [model.add_job() for _ in range(instance.num_jobs)]
    resources = [model.add_machine() for _ in range(instance.num_machines)]

    for job_idx, tasks in enumerate(instance.jobs):
        for task_data in tasks:
            task = model.add_task(job=jobs[job_idx])

            for resource_idx, duration in task_data:
                model.add_mode(task, resources[resource_idx], duration)

    for frm, to in instance.precedences:
        model.add_end_before_start(model.tasks[frm], model.tasks[to])

    return model.data()


def _project_instance_to_data(instance: psplib.ProjectInstance) -> ProblemData:
    """
    Converts a ProjectInstance to a ProblemData object.
    """
    model = Model()

    for resource in instance.resources:
        if resource.renewable:
            model.add_renewable(capacity=resource.capacity)
        else:
            model.add_non_renewable(capacity=resource.capacity)

    for project in instance.projects:
        job = model.add_job(release_date=project.release_date)

        for _ in project.activities:
            model.add_task(job=job)

    for idx, activity in enumerate(instance.activities):
        for mode in activity.modes:
            model.add_mode(
                task=model.tasks[idx],
                resources=model.resources,
                duration=mode.duration,
                demands=mode.demands,
            )

    for idx, activity in enumerate(instance.activities):
        for succ_idx, task_idx in enumerate(activity.successors):
            pred = model.tasks[idx]
            succ = model.tasks[task_idx]

            if activity.delays is None:
                model.add_end_before_start(pred, succ)
            else:
                # RCPSP/max precedence type.
                delay = activity.delays[succ_idx]
                model.add_start_before_start(pred, succ, delay)

    return model.data()
