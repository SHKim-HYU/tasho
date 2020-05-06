"""Environment module for defining objects in the environment."""

class Environment:
    """Docstring for class Environment.

    This should be a description of the Environment class.
    It's common for programmers to give a code example inside of their
    docstring::

        from tasho import Environment
        robot = Environment()

    Here is a link to :py:meth:`__init__`.
    """
    def __init__(self):
        self.obstacles = []
        self.fixed_object = []

    def add_obstacle(obstacle):
        self.obstacles.append(obstacle)

class Obstacle:
    def __init__(self, position = [0,0,0], orientation = [0,0,0]):
        self.position = position
        self.orientation = orientation

class Cylinder(Obstacle):
    def __init__(self, radius = 0.1, height = 0.1, position = [0,0,0], orientation = [0,0,0]):
        self.radius = radius
        self.height = height
        self.position = position
        self.orientation = orientation

class Cube(Obstacle):
    def __init__(self, length = 0.1, position = [0,0,0], orientation = [0,0,0]):
        self.length = length
        self.position = position
        self.orientation = orientation
