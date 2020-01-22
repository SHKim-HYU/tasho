from rockit import Ocp, FreeTime
from rockit.multiple_shooting import MultipleShooting

def world():
    print("This is TASHO, a new MECO Python package")

class OCP:
    def __init__(self, type):
        self.ocp = Ocp(T=FreeTime(10.0))
        self.type = type

    def about(self):
        print(self.type + " is the ocp's type.")
