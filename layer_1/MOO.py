#####################
# Library
#####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import MODEL

#####################
# Class MOO
#####################
class MOO_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_MODEL_instance:"MODEL"):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        # Sampling file of the model
        self.data = pd.DataFrame(columns=["Name", "Type", "Mean", "Standard deviation", "Distribution"])

        self.result = None

    #################################################################################
    ###########           Return the Dataframe of the data               ############
    def __repr__(self) -> str:
        return str(self.data)

    #################################################################################
    #########     Fonction to return the number of sampled elements        ##########
    @property
    def len(self):
        return len(self.data)

