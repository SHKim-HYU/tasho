from tasho import task_prototype_rockit as tp
import copy

class OCPGenerator:

    def __init__(self, task, FreeTime, discretization_settings = None):
        """ 
        A constructor for the OCP generator object.   
        """

        self._solver_settings = {}
        if FreeTime:
            self.tc = tp.task_context(horizon_steps=discretization_settings["horizon_steps"])
        else:
            self.tc = tp.task_context(time = discretization_settings["time_period"], horizon_steps=discretization_settings["horizon_steps"])
        
        self.stages = [self.tc.stages[0]]
        self.stage_tasks = [copy.deepcopy(task)]


    def append_task(self, task, FreeTime, discretization_settings):

        if FreeTime:
            stage = self.tc.create_stage(horizon_steps=discretization_settings["horizon_steps"])
        else:
            stage = self.tc.create_stage(time = discretization_settings["time_period"], horizon_steps=discretization_settings["horizon_steps"])
        self.stages.append(stage)
        self.stage_tasks.append(copy.deepcopy(task))
        return stage

    def _generate_task_ocp(self, stage):

        raise Exception("Not implemented.")


    

