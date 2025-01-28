from numpy.testing import assert_equal

from pyjobshop.Model import Model
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
    SetupTime,
    StartBeforeEnd,
    StartBeforeStart,
    Task,
)
from pyjobshop.Solution import Solution, TaskData


def test_model_to_data():
    """
    Tests that calling ``Model.data()`` returns a correct ProblemData instance.
    """
    model = Model()

    job = model.add_job()
    machine1, machine2 = [model.add_machine() for _ in range(2)]
    task1, task2 = [model.add_task(job=job) for _ in range(2)]

    model.add_mode(task1, machine1, 1)
    model.add_mode(task2, machine2, 2)

    model.add_start_before_start(task1, task2)
    model.add_start_before_end(task1, task2)
    model.add_end_before_end(task1, task2)
    model.add_end_before_start(task1, task2)
    model.add_identical_resources(task2, task1)
    model.add_different_resources(task2, task1)
    model.add_if_then(task2, task1)
    model.add_consecutive(task2, task1, machine1)

    model.add_setup_time(machine1, task1, task2, 3)
    model.add_setup_time(machine2, task1, task2, 4)

    model.set_objective(weight_total_flow_time=1)

    data = model.data()

    assert_equal(data.jobs, [job])
    assert_equal(data.resources, [machine1, machine2])
    assert_equal(data.tasks, [task1, task2])
    assert_equal(
        data.modes,
        [
            Mode(task=0, resources=[0], duration=1),
            Mode(task=1, resources=[1], duration=2),
        ],
    )

    constraints = data.constraints
    assert_equal(constraints.start_before_start, [StartBeforeStart(0, 1)])
    assert_equal(constraints.start_before_end, [StartBeforeEnd(0, 1)])
    assert_equal(constraints.end_before_end, [EndBeforeEnd(0, 1)])
    assert_equal(constraints.end_before_start, [EndBeforeStart(0, 1)])
    assert_equal(constraints.identical_resources, [IdenticalResources(1, 0)])
    assert_equal(constraints.different_resources, [DifferentResources(1, 0)])
    assert_equal(constraints.if_then, [IfThen(1, [0])])
    assert_equal(constraints.consecutive, [Consecutive(1, 0, 0)])
    assert_equal(
        constraints.setup_times, [SetupTime(0, 0, 1, 3), SetupTime(1, 0, 1, 4)]
    )
    assert_equal(data.objective, Objective(weight_total_flow_time=1))


def test_from_data():
    """
    Tests that initializing from a data instance returns a valid model
    representation of that instance.
    """
    data = ProblemData(
        [Job(due_date=1)],
        [Machine(), Renewable(1), NonRenewable(0)],
        [Task(), Task(job=0), Task()],
        modes=[Mode(0, [0], 1), Mode(1, [1], 2), Mode(2, [1], 2)],
        constraints=Constraints(
            start_before_start=[StartBeforeStart(0, 1)],
            start_before_end=[StartBeforeEnd(0, 1)],
            end_before_start=[EndBeforeStart(0, 1)],
            end_before_end=[EndBeforeEnd(0, 1)],
            identical_resources=[IdenticalResources(0, 1)],
            different_resources=[DifferentResources(0, 1)],
            if_then=[IfThen(1, [0])],
            consecutive=[Consecutive(1, 2, 0)],
            setup_times=[
                SetupTime(0, 0, 1, 1),  # machine
                SetupTime(1, 0, 1, 0),  # renewable
                SetupTime(2, 0, 1, 0),  # non-renewable
            ],
        ),
        objective=Objective(
            weight_makespan=2,
            weight_tardy_jobs=3,
            weight_total_tardiness=4,
            weight_total_flow_time=5,
            weight_total_earliness=6,
            weight_max_tardiness=7,
            weight_max_lateness=8,
        ),
    )
    model = Model.from_data(data)
    m_data = model.data()

    assert_equal(m_data.num_jobs, data.num_jobs)
    assert_equal(m_data.num_resources, data.num_resources)
    assert_equal(m_data.num_tasks, data.num_tasks)
    assert_equal(m_data.num_modes, data.num_modes)
    assert_equal(m_data.modes, data.modes)
    assert_equal(m_data.constraints, data.constraints)
    assert_equal(m_data.objective, data.objective)


def test_model_to_data_default_values():
    """
    Tests ``Model.data()`` uses the correct default values.
    """
    model = Model()

    job = model.add_job()
    machine = model.add_machine()
    task = model.add_task(job=job)
    model.add_mode(task, machine, 1)

    data = model.data()

    assert_equal(data.jobs, [job])
    assert_equal(data.resources, [machine])
    assert_equal(data.tasks, [task])
    assert_equal(data.modes, [Mode(task=0, resources=[0], duration=1)])
    assert_equal(data.constraints, Constraints())
    assert_equal(data.objective, Objective(weight_makespan=1))


def test_add_job_attributes():
    """
    Tests that adding a job to the model correctly sets the attributes.
    """
    model = Model()

    job = model.add_job(
        weight=0, release_date=1, due_date=2, deadline=3, name="job"
    )

    assert_equal(job.weight, 0)
    assert_equal(job.release_date, 1)
    assert_equal(job.due_date, 2)
    assert_equal(job.deadline, 3)
    assert_equal(job.name, "job")


def test_add_machine_attributes():
    """
    Tests that adding a machine to the model correctly sets the attributes.
    """
    model = Model()

    machine = model.add_machine(name="machine")
    assert_equal(machine.name, "machine")


def test_add_renewable_resource_attributes():
    """
    Tests that adding a resource to the model correctly sets the attributes.
    """
    model = Model()

    renewable = model.add_renewable(capacity=1, name="resource")

    assert_equal(renewable.capacity, 1)
    assert_equal(renewable.name, "resource")


def test_add_non_renewable_resource_attributes():
    """
    Tests that adding a resource to the model correctly sets the attributes.
    """
    model = Model()

    non_renewable = model.add_non_renewable(capacity=1, name="resource")

    assert_equal(non_renewable.capacity, 1)
    assert_equal(non_renewable.name, "resource")


def test_add_task_attributes():
    """
    Tests that adding a task to the model correctly sets the attributes.
    """
    model = Model()

    task = model.add_task(
        earliest_start=1,
        latest_start=2,
        earliest_end=3,
        latest_end=4,
        fixed_duration=True,
        name="task",
    )

    assert_equal(task.earliest_start, 1)
    assert_equal(task.latest_start, 2)
    assert_equal(task.earliest_end, 3)
    assert_equal(task.latest_end, 4)
    assert_equal(task.fixed_duration, True)
    assert_equal(task.name, "task")


def test_add_mode_attributes():
    """
    Tests that adding a mode to the model correctly sets the attributes.
    """
    model = Model()

    task = model.add_task()
    resources = [model.add_machine() for _ in range(3)]

    mode = model.add_mode(task, resources, duration=1, demands=[1, 2, 3])

    assert_equal(mode.task, 0)
    assert_equal(mode.resources, [0, 1, 2])
    assert_equal(mode.duration, 1)
    assert_equal(mode.demands, [1, 2, 3])


def test_add_mode_single_resource():
    """
    Tests that adding a mode with single resource and demand is correctly set.
    """
    model = Model()

    job = model.add_job()
    machine = model.add_machine()
    task = model.add_task(job=job)

    mode = model.add_mode(task, machine, duration=1, demands=1)
    assert_equal(mode.task, 0)
    assert_equal(mode.resources, [0])
    assert_equal(mode.duration, 1)
    assert_equal(mode.demands, [1])


def test_add_if_then_successor_type_cast():
    """
    Tests that adding an if-then constraint with successors is correctly
    cast to a list, if applicable.
    """
    model = Model()
    task1, task2, task3 = [model.add_task() for _ in range(3)]

    # Single successor is allowed, but this should be cast to a list
    # in the constraint.
    constraint = model.add_if_then(task1, task2)
    assert_equal(constraint.predecessor, 0)
    assert_equal(constraint.successors, [1])

    # Multiple successors are also allwoed.
    constraint = model.add_if_then(task1, [task2, task3])
    assert_equal(constraint.predecessor, 0)
    assert_equal(constraint.successors, [1, 2])


def test_model_attributes():
    """
    Tests that the model attributes are correctly.
    """
    model = Model()

    jobs = [model.add_job() for _ in range(10)]
    resources = [model.add_machine() for _ in range(20)]
    tasks = [model.add_task() for _ in range(30)]
    modes = [model.add_mode(t, [m], 1) for t in tasks for m in resources]

    assert_equal(model.jobs, jobs)
    assert_equal(model.resources, resources)
    assert_equal(model.tasks, tasks)
    assert_equal(model.modes, modes)


def test_model_set_objective():
    """
    Tests that setting the objective changes.
    """
    model = Model()

    # The default objective function is the makespan.
    assert_equal(model.objective, Objective(weight_makespan=1))

    # Now we set the objective function to a weighted combination
    # this should overwrite the previously set objective.
    model.set_objective(
        weight_makespan=1,
        weight_tardy_jobs=2,
        weight_total_tardiness=3,
        weight_total_flow_time=4,
        weight_total_earliness=5,
        weight_max_tardiness=6,
        weight_max_lateness=7,
    )

    assert_equal(model.objective.weight_makespan, 1)
    assert_equal(model.objective.weight_tardy_jobs, 2)
    assert_equal(model.objective.weight_total_tardiness, 3)
    assert_equal(model.objective.weight_total_flow_time, 4)
    assert_equal(model.objective.weight_total_earliness, 5)
    assert_equal(model.objective.weight_max_tardiness, 6)
    assert_equal(model.objective.weight_max_lateness, 7)


def test_solve(solver: str):
    """
    Tests the solve method of the Model class.
    """
    model = Model()

    job = model.add_job()
    machine = model.add_machine()
    tasks = [model.add_task(job=job) for _ in range(2)]

    for task, duration in zip(tasks, [1, 2]):
        model.add_mode(task, machine, duration)

    result = model.solve(solver=solver)

    assert_equal(result.objective, 3)
    assert_equal(result.status.value, "Optimal")


def test_solve_additional_kwargs_initial_solution_fixed(small):
    """
    Tests that Model.solve() correctly takes the initial solution and passes
    additional kwargs to the solver, fixing the solution.
    """
    model = Model.from_data(small)
    init = Solution([TaskData(0, [0], 0, 1), TaskData(1, [0], 3, 5)])
    result = model.solve(
        "ortools",
        initial_solution=init,
        fix_variables_to_their_hinted_value=True,
    )

    # The initial solution has makespan 5 and is fixed, even though this
    # instance admits a better solution with makespan 3.
    assert_equal(result.objective, 5)
    assert_equal(result.status.value, "Optimal")
