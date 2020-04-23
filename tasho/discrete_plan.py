
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

    def solve_task(self, task_name = None, q_init = None, qdot_init = None):
        # TODO: Check if there's any better way to get ndof from tc

        ndof = self.task_list[0].states["q"].size(1)

        if task_name == None:
            task_name = self.task_names[0]
        if q_init == None:
            q_init = [0]*ndof
        if qdot_init == None:
            qdot_init = [0]*ndof

        if type(task_name) is str:
        # if isinstance(task_name, str):
            task_index =  self.task_names.index(task_name)
            tc = self.task_list[task_index]

            # Set parameter's values
            tc.ocp.set_value(tc.parameters["q0"], q_init)
            tc.ocp.set_value(tc.parameters["q_dot0"], qdot_init)

            sol = tc.solve_ocp()

        elif type(task_name) is list:
            sol = []
            for tname in task_name:
                task_index =  self.task_names.index(tname)
                tc = self.task_list[task_index]

                # Set parameter's values
                tc.ocp.set_value(tc.parameters["q0"], q_init)
                tc.ocp.set_value(tc.parameters["q_dot0"], qdot_init)

                sol_tc = tc.solve_ocp()
                sol.append(sol_tc)

                ts, q_sol = sol_tc.sample(tc.states["q"], grid="control")
                ts, qdot_sol = sol_tc.sample(tc.states["q_dot"], grid="control")
                q_init = q_sol[-1,:]
                qdot_init = qdot_sol[-1,:]

        return sol


    def execute_plan(self, q_init = None, qdot_init = None):
        solution_list = []

        # TODO: Check if there's any better way to get ndof from tc
        ndof = self.task_list[0].states["q"].size(1)

        if q_init == None:
            q_init = [0]*ndof
        if qdot_init == None:
            qdot_init = [0]*ndof

        for tc in self.task_list:
            # Set parameter's values
        	tc.ocp.set_value(tc.parameters["q0"], q_init)
        	tc.ocp.set_value(tc.parameters["q_dot0"], qdot_init)

        	sol = tc.solve_ocp()
        	solution_list.append(sol)

        	ts, q_sol = sol.sample(tc.states["q"], grid="control")
        	ts, qdot_sol = sol.sample(tc.states["q_dot"], grid="control")
        	q_init = q_sol[-1,:]
        	qdot_init = qdot_sol[-1,:]

        return solution_list

    def simulate_plan(self, simulator="bullet"):
        # TODO

        if simulator=="bullet":

            for tc in self.task_list:
                print("")
        else:
            print("Not yet implemented")
