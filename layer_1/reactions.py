#####################
# Library
#####################
import pandas as pd
import pint 

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cell_model import MODEL

####################
# Class Reactions
#####################
class Reaction_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_MODEL_instance:"MODEL"):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        self.__unit = "mM/h"
        self.__ureg = self.__class_MODEL_instance.ureg

        self.df = pd.DataFrame(
            columns=["Metabolites", "Reversible", "Enzymatic", "Flux", "Equilibrium constant","Driving force"]
        )

    ################################################################################
    #########           Return the Dataframe of the reactions             ##########
    def __repr__(self) -> str:
        return self.df.to_string()
    

    ################################################################################
    #########        Fonction to return the number of reaction            ##########
    @property
    def len(self):
        return len(self.df)

    #################################################################################
    #########         Fonction to return a list of the reactions           ##########
    @property
    def list(self):
        return(list(self.df.index))

    #################################################################################
    #########                    Gestion of the unit                       ##########
    @property
    def unit(self):
        return(self.__ureg(self.__unit).units)
    
    @unit.setter
    def unit(self, input_unit:str):
        """ Change the unit of the flux """
        try:
            # First we check if the input unit is a flux
            test_unit = 1 * self.__ureg(input_unit)
            test_unit.to("mM/h")

            # Conversion of flux values if the Dataframe didn't have 0 metabolite
            if not self.df.empty:
                # Conversion de l'ancienne unité vers la nouvelle
                old_unit = self.__ureg(self.__unit)
                new_unit = self.__ureg(input_unit)

                # We change the unit of every flux
                self.df["Flux"] = self.df["Flux"].astype(float).apply(
                    lambda x: (x * old_unit).to(new_unit).magnitude
                    )
                
            # Update of the unite
            self.__unit = input_unit

        except pint.UndefinedUnitError:            
            raise ValueError(f"The input unit '{input_unit}' isn't valid, try 'mM/h'")
        except pint.DimensionalityError:
            raise ValueError(f"The input unit '{input_unit}' isn't a reaction rate unit, try 'mM/h'")


    #################################################################################
    #########           Fonction to add a reaction                         ##########
    def add(self, name: str, metabolites={}, reversible=True, enzymatic=True, flux=1.0, k_eq=1.0, DF = 1.0) -> None:
        ### Description of the fonction
        """
        Fonction to add a reaction to the model\n
            If it is already in the model, it change its properties

        Parameters
        ----------
        name        : str
            Name of the reaction\n
        
        metabolites : dict
            Take as keys the names of the metabolites (str) and as value the stoichiometric coefficient (float)\n

        reversible  : bool
            Is the reaction reversible ?\n

        enzymatic   : bool
            Is the reaction enzymatic ?\n
        
        k_eq        : float
            Equilibre constant of the reaction\n

        flux        : float
            Flux of the reaction at the reference state

        """
        # Look if the reaction class was well intialised
        if type(self.df) != type(pd.DataFrame()):
            self.df = pd.DataFrame(
                columns=[
                    "Metabolites",
                    "Reversible",
                    "Enzymatic",
                    "Flux",
                    "Equilibrium constant",
                    "Driving force"])


        # Else, the reaction is add to the model by an add to the DataFrame
        else:
            if name in self.df.index :
                self.change(name, 
                            metabolites=metabolites, 
                            reversible=reversible, 
                            enzymatic=enzymatic, 
                            flux=flux, 
                            k_eq=k_eq, 
                            DF=DF)
            else : 
                # Add the reaction to the reactions dataframe
                self.df.loc[name] = [metabolites, 
                                     reversible, 
                                     enzymatic, 
                                     flux, 
                                     k_eq, 
                                     DF]

                # Add a null columns to the stoichio matrix N
                if name not in self.__class_MODEL_instance.Stoichio_matrix_pd.columns:
                    self.__class_MODEL_instance.Stoichio_matrix_pd[name] = 0.0
                    self.__class_MODEL_instance.Stoichio_matrix_pd = self.__class_MODEL_instance.Stoichio_matrix_pd.copy()

                for meta in list(metabolites.keys()):
                    if meta not in self.__class_MODEL_instance.Stoichio_matrix_pd.index:
                        # If the metabolite is not in the model, we add it
                        self.__class_MODEL_instance.metabolites.add(meta)

                    # Then we add the correct stoichiometric coefficients
                    self.__class_MODEL_instance.Stoichio_matrix_pd.at[meta, name] = self.df.at[name, "Metabolites"][meta]

                # Updating the elasticity matrix
                self.__class_MODEL_instance._update_elasticity()


    #################################################################################
    #########           Fonction to change a reaction                      ##########
    def change(self, name: str, metabolites=None, reversible=None, enzymatic=None, flux=None, k_eq=None, DF = None):
        ### Description of the fonction
        """
        Fonction to change a reaction properties in the model
        
        Parameters
        ----------
        name           : str
            Name of the reaction to change\n
        
        metabolites    : dict
            Dictionnary of the metabolites used in this reaction and their stoichiometric coefficient\n
    
        reversible     : bool
            Specify if the reaction is reversible or not\n

        enzymatic      : bool
            Specify if the reaction is enzymatic or not\n

        flux           : float
            Value of the flux at the reference state\n
    
        k_eq           : float  
            The equilibrium constant of the reaction\n

        DF             : float
            Driving force of the reaction

        """

        if name not in self.df.index:
            raise NameError(f"The name '{name}' is not in the reactions dataframe")

        else:
            # First we look the metabolites input
            if isinstance(metabolites, dict):
                self.df.at[name, "Metabolites"] = metabolites
                self.__class_MODEL_instance.Stoichio_matrix_pd[name] = 0.0

                for meta in list(metabolites.keys()):
                    if meta not in self.__class_MODEL_instance.Stoichio_matrix_pd.index:
                        # If the metabolite is not in the model, we add it
                        self.__class_MODEL_instance.metabolites.add(meta)
                
                    # Then we add the correct stoichiometric coefficients
                    self.__class_MODEL_instance.Stoichio_matrix_pd.at[meta, name] = self.df.at[name, "Metabolites"][meta]

            if reversible != None:
                self.df.at[name, "Reversible"] = reversible
                self.__class_MODEL_instance.elasticity.s._cache_mask_irreversible = None
            
            if enzymatic != None:
                self.df.at[name, "Enzymatic"] = enzymatic
            
            if flux != None:
                self.df.at[name, "Flux"] = flux
            
            if k_eq != None:
                self.df.at[name, "Equilibrium constant"] = k_eq

            if DF != None:
                self.df.at[name, "Driving force"] = DF

    #################################################################################
    #########           Fonction to remove a reaction                      ##########

    def remove(self, name: str) -> None:
        ### Description of the fonction
        """
        Fonction to remove a reaction to the model
        
        Parameters
        ----------
        name        : str
            Name of the reaction to remove
        """

        # Look if the reaction isn't in the model
        if name not in self.df.index:
            True

        # Else, the reaction is remove from the model
        else:
            self.df.drop(name, inplace=True)

            # For a reaction in the stoichiometric matrix
            for reaction in self.__class_MODEL_instance.Stoichio_matrix_pd.columns:
                # If the the reaction is not in the modified reaction dataframe => it was deleted
                if reaction not in self.df.index:
                    self.__class_MODEL_instance.Stoichio_matrix_pd.drop(reaction, axis=1, inplace=True)

            self.__class_MODEL_instance.elasticity.p.df.drop(index=name, inplace=True)
            self.__class_MODEL_instance.elasticity.s.df.drop(index=name, inplace=True)
            # Updating the network
            self.__class_MODEL_instance._update_network

    #################################################################################
    #########           Fonction to add a reaction                         ##########
    def _update(self, name: str, metabolites={}, reversible=True, enzymatic=True, flux=1, k_eq=1.0, DF=1.0) -> None:
        ### Description of the fonction
        """
        Internal function to update the reaction dataframe after a change of the stoichiometric matrix
        
        Parameters
        ----------
        name        : str
            Name of the reaction\n

        metabolites : dict
            Dictionnary that take as keys the names of the metabolites (str) and as value the stoichiometric coefficient (float)\n

        reversible         : bool
            Is the reaction reversible ?\n

        enzymatic         : bool
            Is the reaction enzymatic ?\n

        flux              : float
            Value of the flux \n

        k_eq        : float
            Equilibre constant of the reaction\n

        DF          : float
            Driving force of the reaction

        """
        # Look if the reaction class was well intialised
        if not isinstance(self.df, pd.DataFrame):
            self.df = pd.DataFrame(columns=["Metabolites", "Reversible", "Enzymatic", "Flux", "Equilibrium constant", "Driving force"]
            )

        # Look if the reaction is already in the model
        if name not in self.df.index:
            self.df.loc[name] = [metabolites, reversible, flux, k_eq, DF]

            # We check the stoichiometric coefficient link to this reaction in order to automatically add them to the matrix
            for meta in list(self.df.loc[name, "Metabolites"].keys()):
                if meta not in self.__class_MODEL_instance.Stoichio_matrix_pd.index:
                    # If the metabolite is not in the model, we add it
                    self.__class_MODEL_instance.metabolites._update(meta)
