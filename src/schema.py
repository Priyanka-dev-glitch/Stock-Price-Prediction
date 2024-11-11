from enum import Enum


class Model_List(str, Enum):
    Linear_reg = "Linear Regression Model"
    K_nn = "k-Nearest Neighbour Model"
    Lstm = "LSTM Model"
    Default = ""

    @classmethod
    def List_params(cls):
        role_names = [member.value for role, member in cls.__members__.items()]
        return role_names


