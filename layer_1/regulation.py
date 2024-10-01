#####################
# Library
#####################
import pandas as pd
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import MODEL

#####################
# Class Regulation
#####################
class Regulation_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_MODEL_instance:"MODEL"):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        self.df = pd.DataFrame(
            columns=[
                "Regulated flux",
                "Regulator",
                "Coefficient of regulation",
                "Type regulation",
                "Activated"
            ]
        )

    #################################################################################
    #########           Return the Dataframe of the            ##########
    def __repr__(self) -> str:
        return str(self.df)

    #################################################################################
    #########        Fonction to return the shape of the matrix            ##########
    @property
    def len(self):
        return len(self.df.shape)

    #################################################################################
    #########           Fonction to add a regulation                       ##########
    def add(
        self, name: str, regulated: str, regulator: str, coefficient=0.5, allosteric=True, activated = True, **kwargs):
        ### Description of the fonction
        """
        Fonction to add a regulation to the model
        
        Parameters
        ----------

        regulated       : str
            Name of regulated flux\n

        regulator       : str
            Name of the metabolite that regulate\n

        coefficient     : foat
            Coefficient of regulation, coef > 0 => activation, coef < 0 => inihibition\n

        allosteric      : str
            Specify the type of reaction, True => allosteric, False => transcriptional

        activated       : bool
            Is the regualation arrow activated ? 
        """
        if allosteric == True:
            type_regulation = "allosteric"
        else:
            type_regulation = "transcriptional"

        # Look if the metabolite class was well intialised
        if type(self.df) != type(pd.DataFrame()):
            self.df = pd.DataFrame(
                columns=[
                    "Regulated flux",
                    "Regulator",
                    "Coefficient of regulation",
                    "Type regulation",
                    "Activated"
                ]
            )

        # Look if the regulation is already in the regulation dataframe
        elif name in self.df.index:
            raise NameError(
                f"The name of the regulation '{name}' is already in the regulation dataframe !"
            )

        # Look if the regulated flux is in the model
        elif regulated not in self.__class_MODEL_instance.reactions.df.index:
            raise NameError(
                f'The reaction "{regulated}" is not in the reaction dataframe !'
            )

        #  Look if the regulator metabolite is in the model
        elif regulator not in self.__class_MODEL_instance.metabolites.df.index:
            raise NameError(
                f'The metabolite "{regulator}" is not in the metabolite dataframe !'
            )
        elif regulator not in self.__class_MODEL_instance.N_without_ext.index:
            raise NameError(
                f'The metabolite "{regulator}" is in the metabolite dataframe but is external !'
            )

        if not isinstance(coefficient, (int,float)) :
            raise TypeError(f"The input argument 'coefficient' must be a float, not a {type(coefficient)} !\n")
        if not isinstance(allosteric, bool) :
            raise TypeError(f"The input argument 'allosteric' must be a bool, not a {type(allosteric)} !\n")
        if not isinstance(activated, bool) :
            raise TypeError(f"The input argument 'activated' must be a bool, not a {type(activated)} !\n")

        # Else it's allright :D
        self.df.loc[name] = [regulated, regulator, coefficient, type_regulation, activated]

        # If the regulation arrow is activated, we modifiy the elasticity of the model
        if activated == True :

            # If the regulation is allosteric
            if allosteric == True :
                self.__class_MODEL_instance.elasticity.s.df.at[
                    regulated, regulator
                ] += coefficient

            # Else, it is a transcriptionnal regulation
            else:
                # name of the enzyme linked to this regulation
                enzyme = "enzyme_" + name

                # We concidere now this enzyme as a metabolite
                self.__class_MODEL_instance.metabolites.add(name=enzyme)
                self.__class_MODEL_instance.reactions.add(
                    name="creation_" + name, metabolites={enzyme: 1}
                )
                self.__class_MODEL_instance.reactions.add(
                    name="destruction_" + name, metabolites={enzyme: -1}
                )
                self.__class_MODEL_instance.elasticity.s.df.at[
                    "creation_" + name, regulator
                ] += coefficient

                default_name = "para_trans_" + name
                name_new_para = kwargs.get('name_parameter', default_name)

                self.__class_MODEL_instance.parameters.add(name=name_new_para)

            self.__class_MODEL_instance._update_elasticity()



    #################################################################################
    #########        Fonction to remove a regulation arrow                ###########
    def remove(self, name: str) -> None:
        ### Description of the fonction
        """
        Fonction to remove regulation arrow
        
        Parameters
        ----------

        name        : str
            Name of the regulation arrow to remove
        """

        self.inactivate(name)

        self.df.drop(name)



    #################################################################################
    #########           Fonction to change a regulation coefficient        ##########
    def change_coeff(self, name_regu: str, new_coeff: float) -> None:
        ### Description of the fonction
        """
        Fonction to change the coefficient of a regulation effect
        
        Parameters
        ----------

        name_regu       : str
            Name of regulation effect to change\n

        new_coeff       : float
            New value of the regulation coefficient

        """
        # Check if the regulation name is in the dataframe
        if name_regu not in self.df.index:
            raise NameError(
                f"The regulation name '{name_regu}' is not in the regulation dataframe"
            )
        

        else:
            regulated = self.df.at[name_regu, "Regulated flux"]
            regulator = self.df.at[name_regu, "Regulator"]
            # if it is an allosteric/direct regulation
            if self.df.at[name_regu, "Type regulation"] == "allosteric":
                # Soustraction of the old value and addition of the new one on the E_s matrix
                self.__class_MODEL_instance.elasticity.s.df.at[
                    regulated, regulator
                ] += (new_coeff - self.df.at[name_regu, "Coefficient of regulation"])

            # Case of a transcriptional/undirect regulation
            else:
                # Soustraction of the old value and addition of the new one on the E_s matrix
                self.__class_MODEL_instance.elasticity.s.df.at[
                    "creation_" + name_regu, regulator
                ] += (new_coeff - self.df.at[name_regu, "Coefficient of regulation"])

            # Attribution of the new value to the coeff
            self.df.at[name_regu, "Coefficient of regulation"] = new_coeff



    #################################################################################
    #########        Fonction to activate a regulation arrow               ##########

    def activate(self, name: str) -> None:
        ### Description of the fonction
        """
        Fonction to activate a regulation arrow
        
        Parameters
        ----------

        name        : str
            Name of the regulation name to activate
        """

        # Look if the regulation is in the model
        if name not in self.df.index:
            raise NameError(
                f"The regulation {name} is not in the regulation dataframe, please enter a valide name \n"
            )
        
        regulated = self.df.at[name, "Regulated flux"]
        regulator = self.df.at[name, "Regulator"]
        coeff = self.df.at[name, "Coefficient of regulation"]
        # If the regulation arrows is initialy inactivated, we modifiy the elasticity
        if self.df.at[name, "Activated"] == False : 
            
            # Case where it is an alosteric regulation
            if self.df.at[name, "Type regulation"] == "allosteric" :
                self.__class_MODEL_instance.elasticity.s.df.at[regulated, regulator] += coeff
            
            # Case where itis a transcriptionnal regulation
            elif self.df.at[name, "Type regulation"] == "transcriptional" :
            # name of the enzyme linked to this regulation
                enzyme = "enzyme_" + name

                # We concidere now this enzyme as a metabolite
                self.__class_MODEL_instance.metabolites.add(name=enzyme)
                self.__class_MODEL_instance.reactions.add(
                    name="creation_" + name, metabolites={enzyme: 1}
                )
                self.__class_MODEL_instance.reactions.add(
                    name="destruction_" + name, metabolites={enzyme: -1}
                )
                self.__class_MODEL_instance.elasticity.s.df.at[
                    "creation_" + name, regulator
                ] += coeff


        self.df.at[name, "Activated"] = True

        # Then we update the rest of the model
        self.__class_MODEL_instance._update_network()




    #################################################################################
    #########        Fonction to inactivate a regulation arrow            ##########

    def inactivate(self, name: str) -> None:
        ### Description of the fonction
        """
        Fonction to inactivate a regulation arrow
        
        Parameters
        ----------

        name        : str
            Name of the regulation name to inactivate
        """

        # Look if the regulation is in the model
        if name not in self.df.index:
            raise NameError(
                f"The regulation {name} is not in the regulation dataframe, please enter a valide name \n"
            )
        
        regulated = self.df.at[name, "Regulated flux"]
        regulator = self.df.at[name, "Regulator"]
        coeff = self.df.at[name, "Coefficient of regulation"]

        # If the regulation arrows is initialy activated, we modifiy the elasticity
        if self.df.at[name, "Activated"] == True : 
            
            # Case where it is an alosteric regulation
            if self.df.at[name, "Type regulation"] == "allosteric" :
                # We substract the coeff
                self.__class_MODEL_instance.elasticity.s.df.at[regulated, regulator] -= coeff
            
            # Case where itis a transcriptionnal regulation
            elif self.df.at[name, "Type regulation"] == "transcriptional" :
                # name of the enzyme linked to this regulation
                enzyme = "enzyme_" + name

                # We remove the enzyme that where considere like a metabolite
                self.__class_MODEL_instance.metabolites.remove(enzyme)
                self.__class_MODEL_instance.reactions.remove(name="creation_" + name)
                self.__class_MODEL_instance.reactions.remove(name="destruction_" + name)


        self.df.at[name, "Activated"] = False

        # Then we update the rest of the model
        self.__class_MODEL_instance._update_network()


    #################################################################################
    #########      Fonction to read a file of regulation database          ##########

    def read_file(self, file_path: str) -> None:
        ### Description of the fonction
        """
        Fonction to read a file of regulation database
        
        Parameters
        ----------

        name        : str
            File_path of the regulation database (SBTab format)
        """
        import sbtab

        filename = file_path.split('/')[-1]
        St = sbtab.SBtab.read_csv(filepath=file_path, document_name=filename)

        table = St.sbtabs[0] 

        for reg in table.value_rows :
            
            meta = reg[0]
            flux = reg[1]
            type_reg = reg[2]
            print(f"{meta} ; {flux} ; {type_reg}\n")
            allosteric_list = ["direct", "alosteric"]
            if type_reg.split(" ")[0].lower() in allosteric_list  :
                allosteric = True
            else :
                allosteric = False

            coeff = 1
            if type_reg.split(" ")[1] == "inhibition" :
                    coeff = -1

            self.__class_MODEL_instance.regulations.add(name = f"{type_reg} {meta} -> {flux}", regulated=flux, regulator=meta, coefficient=coeff,allosteric=allosteric)
