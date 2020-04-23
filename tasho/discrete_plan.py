
"""Discrete plan module for a plan to execute."""

from casadi import vertcat


class DiscretePlan:
    """Docstring for class Discrete Plan.

    This should be a description of the Discrete Plan class.
    It's common for programmers to give a code example inside of their
    docstring::

        from tasho import DiscretePlan
        plan = DiscretePlan()
        plan.add_task(tc)
        sol_list = plan.execute_plan()

    Here is a link to :py:meth:`__init__`.
    """


    def __init__(self):
        """Instantiate a Discrete Plan.

        :param name: Robots name to load functions.
        :type name: string

        """
        # self.name = name
        self.robot_list = None
        self.gravity = vertcat(0,0,-9.81)
        self.task_list = []
        self.task_names = []

    def add_task(self, task, name = None):
        self.task_list.append(task)
        if name == None:

            self.task_names.append("task_"+str(len(self.task_list)))
        else:
            self.task_names.append(name)

    def print_tasks(self):
        print(self.task_names)

    def execute_plan(self):
        solution_list = []
        for tc in self.task_list:
            sol = tc.solve_ocp()
            solution_list.append(sol)
            ts, q_sol = sol.sample(tc.states["q"], grid="control")
            print(q_sol)
            # print(tc.states)
        return solution_list

    def simulate_plan(self):
        for tc in self.task_list:
            sol = tc.solve_ocp()
        # TODO
