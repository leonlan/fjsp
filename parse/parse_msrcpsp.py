# 11 Jan 2025
# Scratch: OR-Tools doesn't work well. CP Optimizer does, but it's probably
# just using CPLEX under the hood.
# TODO: write a new OR-Tools model that uses alternative constraints.
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Union
from pyjobshop import Model
from pyjobshop.plot import plot_machine_gantt, plot_task_gantt
from itertools import pairwise
import matplotlib.pyplot as plt


@dataclass
class Activity:
    duration: int
    successors: list[int]
    skill_requirements: list[int]


@dataclass
class Resource:
    skills: list[int]


@dataclass
class ProjectInstance:
    resources: list[Resource]
    activities: list[Activity]

    @property
    def num_resources(self):
        return len(self.resources)

    @property
    def num_activities(self):
        return len(self.activities)

    @property
    def skills(self):
        return sorted(
            {
                skill
                for activity in self.activities
                for skill in activity.skill_requirements
            }
        )


def parse_msrcpsp(loc: Union[str, Path]) -> ProjectInstance:
    """
    Parses a MS-RCPSP formatted instance from Snauwaert and Vanhoucke (2023).
    """
    with open(loc, "r") as fh:
        lines = iter(line.strip() for line in fh.readlines() if line.strip())

    # Project module.
    next(lines)
    num_activities, num_resources, num_skills, num_levels = map(
        int, next(lines).split()
    )
    deadline = int(next(lines))
    deadline_skill_level_req = int(next(lines))

    durations = []
    successors = []
    for _ in range(num_activities):
        line = iter(map(int, next(lines).split()))
        durations.append(int(next(line)))

        # Succesors are 1-indexed.
        num_successors = int(next(line))
        successors.append([int(next(line)) - 1 for _ in range(num_successors)])

    # Workforce module.
    next(lines)
    res_bool_skills = [
        map(int, next(lines).split()) for _ in range(num_resources)
    ]

    # Workforce module with skill levels.
    next(lines)
    res_skill_levels = [next(lines) for _ in range(num_resources)]

    # Skill requirements module.
    next(lines)
    skill_reqs = [
        list(map(int, next(lines).split())) for _ in range(num_activities)
    ]

    # Skill level requirements module.
    next(lines)
    skill_level_reqs = [next(lines) for _ in range(num_skills)]

    # All other lines are for variants that I'm ignoring for now.

    # Convert to project instance objects.
    resources = [
        Resource([idx for idx, skill in enumerate(skills) if skill == 1])
        for skills in res_bool_skills
    ]
    activities = [
        Activity(
            duration=durations[idx],
            successors=successors[idx],
            skill_requirements=skill_reqs[idx],
        )
        for idx in range(num_activities)
    ]

    return ProjectInstance(resources, activities)


if __name__ == "__main__":
    loc = "tmp/MSLIB/MSLIB1/Instances1"
    path = Path(loc)

    for idx in [2]:
        instance_loc = path / f"MSLIB_Set1_{idx}.msrcp"
        instance = parse_msrcpsp(instance_loc)

        model = Model()

        skills2resources = defaultdict(list)
        for res in instance.resources:
            resource = model.add_renewable(capacity=1)
            for skill in res.skills:
                skills2resources[skill].append(resource)

        dummy = model.add_renewable(capacity=100)

        starts = []
        ends = []

        for idx, activity in enumerate(instance.activities):
            name = f"task_{idx}"
            job = model.add_job(name=name)

            # Start and end of each task, with dummy modes.
            start = model.add_task(job=job, name=name + "_start")
            end = model.add_task(job=job, name=name + "_end")

            starts.append(start)
            ends.append(end)

            model.add_mode(start, dummy, 0, demands=0)
            model.add_mode(end, dummy, 0, demands=0)

            # Create a new task for each skill.
            task_skills = []
            for skill, num_required in enumerate(activity.skill_requirements):
                for num_req in range(num_required):
                    task_skill = model.add_task(
                        job=job, name=name + f"_skill_{skill}_{num_req}"
                    )
                    task_skills.append(task_skill)

                    # Skill task is between start and end.
                    model.add_end_before_start(start, task_skill)
                    model.add_end_before_start(task_skill, end)

                    # For each resource with this skill, add a mode.
                    for resource in skills2resources[skill]:
                        model.add_mode(
                            task_skill, resource, activity.duration, demands=1
                        )

            # All skill tasks must be processed in parallel.
            for task1, task2 in pairwise(task_skills):
                model.add_start_before_start(task1, task2)
                model.add_start_before_start(task2, task1)
                model.add_end_before_end(task1, task2)
                model.add_end_before_end(task2, task1)
                model.add_different_resource(task1, task2)

        for idx, activity in enumerate(instance.activities):
            for succ in activity.successors:
                first = ends[idx]
                second = starts[succ]
                model.add_end_before_start(first, second)

        display = True
        time_limit = 600
        solver = "cpoptimizer"
        # solver = "ortools"
        num_workers = 1
        result = model.solve(
            display=display,
            time_limit=time_limit,
            solver=solver,
            num_workers=num_workers,
            SearchType="DepthFirst",
        )

        res = (
            instance_loc.stem,
            result.objective,
            str(result.status.value),
            round(result.runtime, 2),
        )
        # plot_machine_gantt(result.best, model.data(), plot_labels=True)
        # plt.show()

        # break
        # with open("other2.txt", "a") as fh:
        #     fh.write(" ".join(map(str, res)) + "\n")

        # check_solution(result.best, instance)

        # with open(f"tmp/aslib-sols/{instance_loc.name}", "w") as fh:
        #     fh.write(str(result.objective) + "\n")
        #     for idx, task in enumerate(result.best.tasks):
        #         fh.write(f"{idx},{task.present},{task.start},{task.end}\n")
