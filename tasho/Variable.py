class Variable:

    """
    A class for Variable data-type.
    """

    def __init__(self, name, type, shape, tc, id = None):
        """
        Constructs the variable object.

        :param name: name of the symbolic variable
        :type name: String

        :param type: type of the symbolic variable. Must be 'state', 'control', 'parameter' or 'variable'.
        :type type: String

        :param shape: (number of rows, number of columns) of the created variable.
        :param type: Integer tuple of size 2.

        :param tc: The OCP wrapper (e.g. task_prototype_rockit) for creating the variables.
        :type tc: task_prototype_rockit object
        
        """
        
        assert isinstance(name, str), "Wrong type " + str(type(name)) + " passed as an argument for variable name." 
        self._name = name

        if id != None:
            self._id = id + '_' + self.name
        else:
            self._id = self.name
        
        assert isinstance(type, str), "Must pass String. \n Instead you passed " + str(type(type))
        assert type == 'state' or type == 'variable' or type == 'control' or type == 'parameter', "Unrecognized variable type requested."
        
        if type == 'state':
            self._x = tc.create_state(self.id, shape)

        elif type == 'control':
            self._x = tc.create_control(self.id, shape)

        elif type == 'parameter':
            self._x = tc.create_parameter(self.id, shape)

        elif type == 'variable':
            self._x = tc.create_variable(self.id, shape)

        self._children_variables = set(self.x)


    @property
    def x(self):
        return self._x

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def children_variables(self):
        return self._children_variables


