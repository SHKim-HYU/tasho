class Variable:

    """
    A class for Variable data-type.
    """

    def __init__(self, name, mid, type, shape):
        """
        Constructs the variable object.

        :param name: name of the symbolic variable. Concatenating the mid and name should lead to a unique id for the variable.
        :type name: String

        :param type: type of the symbolic variable. Must be 'state', 'control', 'parameter' or 'variable'.
        :type type: String

        :param shape: (number of rows, number of columns) of the created variable.
        :param type: Integer tuple of size 2.

        :param mid: Meta-id that specifies type of the variable.
        :type mid: String
        
        """
        
        assert isinstance(name, str), "Wrong type " + str(type(name)) + " passed as an argument for variable name." 
        self._name = name
        self._mid = mid
        self._uid = mid + '_' + name

        assert isinstance(type, str), "Must pass String. \n Instead you passed " + str(type(type))
        assert type == 'state' or type == 'variable' or type == 'control' or type == 'parameter', "Unrecognized variable type requested."
        
        if type == 'state':
            self._x = tc.create_state(self.uid, shape)

        elif type == 'control':
            self._x = tc.create_control(self.uid, shape)

        elif type == 'parameter':
            self._x = tc.create_parameter(self.uid, shape)

        elif type == 'variable':
            self._x = tc.create_variable(self.uid, shape)

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
    def uid(self):
        return self._uid