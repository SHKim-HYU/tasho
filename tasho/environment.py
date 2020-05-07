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
        self.objects = {}

    def add_object(self, object, name = None):
        if name is None:
            name = "object_"+str(len(self.objects)+1)
        self.objects[name] = object

    def print_objects(self):
        print([key for key in self.objects.keys()])

    def set_in_world_simulator(self, world_simulator):
        for key, value in self.objects.items():
            #TODO: Set what to do depending on what type of object it is
            _className = value.__class__.__name__

            if _className == "Cube":
                print("*********** THIS IS A CUBE ************")
                objectID = world_simulator.add_object_urdf(position = value.position, orientation = value.orientation, urdf = value.urdf, fixedBase = value.fixed, globalScaling = value.length)
            elif _className == "Box":
                print("*********** THIS IS A BOX ************")
                objectID = world_simulator.add_object_urdf(position = value.position, orientation = value.orientation, urdf = value.urdf, fixedBase = value.fixed, globalScaling = value.height)

class Object:
    def __init__(self, position = [0,0,0], orientation = [0,0,0], urdf = None, fixed = False):
        self.position = position
        self.orientation = orientation
        self.urdf = urdf
        self.fixed = fixed

class Sphere(Object):
    def __init__(self, radius = 0.1, position = [0,0,0], urdf = None, fixed = False):
        self.radius = radius
        self.position = position
        self.urdf = urdf
        self.fixed = fixed
        # self.type = "Sphere"

class Cylinder(Object):
    def __init__(self, radius = 0.1, height = 0.1, position = [0,0,0], orientation = [0,0,0], urdf = None, fixed = False):
        self.radius = radius
        self.height = height
        self.position = position
        self.orientation = orientation
        self.urdf = urdf
        self.fixed = fixed

class Cube(Object):
    def __init__(self, length = 0.1, position = [0,0,0], orientation = [0,0,0], urdf = "models/objects/cube_small.urdf", fixed = False):
        self.length = length
        self.position = position
        self.orientation = orientation
        self.urdf = urdf
        self.fixed = fixed

class Box(Object):
    def __init__(self, length = 0.1, depth = 0.1, height = 0.1 , position = [0,0,0], orientation = [0,0,0], urdf = None, fixed = False):
        self.length = length
        self.depth = depth
        self.height = height
        self.position = position
        self.orientation = orientation
        self.urdf = urdf
        self.fixed = fixed
