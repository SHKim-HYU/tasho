import casadi as cs
import tasho #import Expression

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
        self._parents = []
        self._parent_uid = []

        assert isinstance(type, str), "Must pass String. \n Instead you passed " + str(type(type))
        assert type == 'state' or type == 'variable' or type == 'control' or type == 'parameter' or type == 'magic_number', "Unrecognized variable type requested."
        self._type = type
        self._shape = shape

        #If the type is a magic number, store the value as a number
        if type == 'magic_number':
            self._x = None
        else:
            self._x = cs.MX.sym(self._uid, *shape)
    
    # setting the derivative of state variable
    def set_der(self, expr):
        
        assert self.type == 'state', "Attempting to set derivative to a non-state variable"
        assert isinstance(expr, tasho.Expression.Expression)
        self._der = expr.uid

    @property
    def x(self):
        return self._x

    

    @property
    def mid(self):
        return self._mid

    @property
    def name(self):
        return self._name

    @property 
    def uid(self):
        return self._uid

    @property
    def shape(self):
        return self._x.shape

    @property 
    def type(self):
        return self._type

    @property
    def parents(self):
        return self._parents

    @property 
    def parent_uid(self):
        return self._parent_uid

    @property
    def der(self):
        return self._der

    