# Specialized OR-Tools model for the MS-RCPSP.
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from ortools.sat.python.cp_model import CpModel, CpSolver, IntervalVar, IntVar
from parse_msrcpsp import parse_msrcpsp
from typing_extensions import IntVar


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


def parse_msrcpsp_correia(loc: Union[str, Path]) -> ProjectInstance:
    """
    Parses a MS-RCPSP formatted instance from Snauwaert and Vanhoucke (2023).
    """
    with open(loc, "r") as fh:
        lines = iter(line.strip() for line in fh.readlines() if line.strip())

    # Project module.
    num_activities, num_resources, num_skills = map(int, next(lines).split())

    # Workforce module.
    res_bool_skills = [
        map(int, next(lines).split()) for _ in range(num_resources)
    ]

    # Activities module.
    durations = []
    skill_reqs = []
    successors = []
    for _ in range(num_activities):
        line = iter(map(int, next(lines).split()))
        durations.append(int(next(line)))

        reqs = [int(next(line)) for _ in range(num_skills)]
        skill_reqs.append(reqs)

        # Succesors are 1-indexed.
        num_successors = int(next(line))
        successors.append([int(next(line)) - 1 for _ in range(num_successors)])

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


@dataclass
class TaskVar:
    start: IntVar
    duration: IntVar
    end: IntVar
    interval: IntervalVar


def new_model(instance):
    H = 1000

    model = CpModel()

    task_vars = []
    for idx in range(instance.num_activities):
        start = model.new_int_var(0, H, name="")
        duration = model.new_constant(instance.activities[idx].duration)
        end = model.new_int_var(0, H, name="")
        interval = model.new_interval_var(start, duration, end, name="")
        task_vars.append(TaskVar(start, duration, end, interval))

    # Skill variables: (activity, skill, resource) key boolean variables.
    skill_vars = {}
    for idx in range(instance.num_activities):
        for skill in instance.skills:
            for resource in range(instance.num_resources):
                var = model.new_bool_var(name="")
                skill_vars[idx, skill, resource] = var

    # Variable: use resource for activity.
    res_act_vars = [
        [model.new_bool_var(name="") for _ in range(instance.num_activities)]
        for _ in range(instance.num_resources)
    ]

    # Constraint: only consider skill variables if activity requires the skill
    # or if the resource has the skill.
    for (activity, skill, resource), var in skill_vars.items():
        res_skills = instance.resources[resource].skills
        model.add(var <= int(skill in res_skills))
        model.add(var <= res_act_vars[resource][activity])

    # Constraint: select exactly as many skills as required.
    for idx, activity in enumerate(instance.activities):
        for skill, num_required in enumerate(activity.skill_requirements):
            expr = [
                skill_vars[idx, skill, resource]
                for resource in range(instance.num_resources)
            ]
            model.add(sum(expr) == num_required)

    # Each resource has T optional intervals, one for each task.
    resource_vars = [
        [
            model.new_optional_interval_var(
                start=task_vars[idx].start,
                size=task_vars[idx].duration,
                end=task_vars[idx].end,
                is_present=res_act_vars[resource][idx],
                name="",
            )
            for idx in range(instance.num_activities)
        ]
        for resource in range(instance.num_resources)
    ]

    # No overlap for each resource.
    for idx in range(instance.num_resources):
        model.add_no_overlap(resource_vars[idx])

    # Precedence constraints.
    for idx, activity in enumerate(instance.activities):
        for succ in activity.successors:
            model.add(task_vars[idx].end <= task_vars[succ].start)

    model.minimize(task_vars[-1].end)

    return model


if __name__ == "__main__":
    # for idx in range(1, 200):
    #     loc = "tmp/MSLIB/Correia(2012)/Instances"
    #     path = Path(loc)
    #     instance_loc = path / f"Correia_Set_{idx}.msrcp"
    #     instance = parse_msrcpsp_correia(instance_loc)

    for idx in range(1, 2001):
        loc = "tmp/MSLIB/MSLIB1/Instances1"
        path = Path(loc)
        instance_loc = path / f"MSLIB_Set1_{idx}.msrcp"
        instance = parse_msrcpsp(instance_loc)

        model = new_model(instance)

        display = False
        time_limit = 30
        num_workers = 8
        params = {
            "max_time_in_seconds": time_limit,
            "log_search_progress": display,
            # 0 means using all available CPU cores.
            "num_workers": num_workers if num_workers is not None else 0,
        }
        cp_solver = CpSolver()
        for key, value in params.items():
            setattr(cp_solver.parameters, key, value)

        status_code = cp_solver.solve(model)
        status = cp_solver.status_name(status_code)
        objective_value = cp_solver.objective_value
        objective_bound = cp_solver.best_objective_bound

        with open("skills.txt", "a") as fh:
            print(
                instance_loc.stem,
                objective_value,
                objective_bound,
                status,
                cp_solver.wall_time,
                file=fh,
            )

        print(
            instance_loc.stem,
            objective_value,
            objective_bound,
            status,
            round(cp_solver.wall_time, 2),
        )
