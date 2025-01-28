from dataclasses import dataclass


@dataclass
class TaskData:
    """
    Stores scheduling data related to a task.

    Parameters
    ----------
    mode
        The selected mode.
    resources
        The selected resources.
    start
        The start time.
    end
        The end time.
    present
        Whether the task is present in the solution.
    """

    mode: int
    resources: list[int]
    start: int
    end: int
    present: bool = True


class Solution:
    """
    Solution to the problem.

    Parameters
    ----------
    tasks
        The list of scheduled tasks.
    """

    def __init__(self, tasks: list[TaskData]):
        self._tasks = tasks

    @property
    def tasks(self) -> list[TaskData]:
        """
        Returns the list of tasks and its scheduling data.
        """
        return self._tasks

    def __eq__(self, other) -> bool:
        return self.tasks == other.tasks

    @property
    def makespan(self) -> int:
        """
        Returns the makespan of the solution.
        """
        return max(task.end for task in self.tasks)
