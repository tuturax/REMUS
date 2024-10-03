# META-REMUS
## _Model for Entropy Transduction Analysis for REsponses of the Metabolism under Uncertain States_

========================================
Python code developed by Arthur Lequertier as part of a thesis supervised by [Wolfram Liebermeister](https://github.com/liebermeister) and funded by the project [AMN](https://jfaulon.com/artificial-metabolic-networks-an-anr-funded-project/) (N°ANR-21CE45-0021-02)

This python module aims to create a model of a metabolic network subject to sources of uncertainty, and to approximate the uncertain response states of the network's internal elements.

Below are the steps for creating and studying a REMUS model.

## Installation (pip installer)

The model can be  installed

```sh
pip install
```
## Packages used on Python 3.10
#### **Necessary** for the computation and data storage
- [Pandas](https://pandas.pydata.org/docs/getting_started/install.html) (1.5)
- [Numpy](https://numpy.org/install/) (1.23)
- [scipy](https://scipy.org/install/) (1.9)
- [sympy](https://www.sympy.org/en/download.html) (1.11)

#### To read files containing metabolic models
- [libsbml](https://sbml.org/software/libsbml/libsbml-docs/installation/)  (5.19)  # For read SBML file
- [sbtab](https://www.sbtab.net/sbtab/default/downloads.html) (1.0)
- [cobra](https://opencobra.github.io/cobrapy/) (0.26)  # To write a cobra fill from the current model

#### For display the result
- [matplotlib](https://matplotlib.org/stable/users/getting_started/) (3.6)
- [tkinter](https://docs.python.org/3/library/tkinter.html) (8.6)  # for graphic interface
- [escher](escher) (1.7)  # For Escher-map


## Import the MODEL Class
```python
    from main import MODEL
``` 

## Creation of a model instance
```python
    model = MODEL()
``` 

## Fill this model
While your model instance is set, you have different possibilities to fill it.

1. Creat from scratch with .add() methods
```python
    model.metabolites.add(name="meta_name", mean=1.0, external=False)
    model.reactions.add(name="reaction_name", mean=1.0, reversible=True)
``` 
2. Use an existing matrix and attribuate it to the stoichiometric matrix. If it is a Pandas dataframe, it with different metabolite or reactions, it reacreate a model, if it is a Pandas dataframe with same index and colimn or a Numpy matrix with the same size, it will just modify the coefficents. 
```python
    model.N = dataframe_N
``` 
3. Using a predefined function for simple network
```python
    model.creat_linear(n=4)
    model.creat_repressilator(n=3) # Coming soon
``` 
4. Launch an existing metabolic model
```python
    model.read_SBML(filepath)
    model.read_CSV(filepath)
    model.read_SBtab(filepath)
``` 
- Systems Biology Markup Language ([SBML](https://sbml.org/)) is a markup language readed thanks to the dedicated Python library [libsbml](https://sbml.org/software/libsbml/libsbml-docs/installation/)
- model.read_CSV() read an Excel like file that have as first row the name of the reaction and as first colomn the metabolite names. At the interseaction, it is the stoichiometric coefficents
- [SBTab](https://www.sbtab.net/) is a conventioned file for structured data tables in Systems Biology.

## Set the sources of uncertainties
Once the model is build, it represent a network at a reference state. It is now necessary to add sources of uncertainties to it, called parameters. The most important input value is the standard deviation of this parameter, because it represent the inital level of uncertainty that will spread throught the network.
```python
    model.parameters.add(name="name_para", mean=1., Standard_deviation=1.)
``` 
But there is also the possibility to use methods that add parameters automatically
```python
    # First, for every reacton, we creat an enzyme that is linked to it
    model.enzymes.add_to_all_reaction()
    # And then we add this enzyme as parameter (internal sources of uncertainties)
    model.parameters.add_enzymes()
    
    # We add every external metabolite as parameters (external source of uncertainties)
    model.parameters.add_externals()
``` 

## Add dynamic interactions
Elasticity coefficients represent the local influence of metabolites (only internal ones) or parameters on fluxes. They are regrouped in 2 matrices $$\varepsilon_s$$ and $$\varepsilon_p$$ 
Like the stoichiometric matrix, you have different possibilities to fill both of them
1. Change the coeffient 1 by 1
```python
    model.elasticity.s.change(flux_name="reaction_name", metabolite_name="meta_name", value=1.)
    model.elasticity.p.change(flux_name="reaction_name", parameter_name="para_name", value=1.)
``` 
2. Set an existing matrix with the same size that the previous one (Numpy or Pandas dataframe)
```python
    model.elasticity.s.df = dataframe
    model.elasticity.p.df = np_array_2D
``` 
3. Use existing function
By assuming half-saturated enzymic reaction $$=> \varepsilon_s = -\frac{1}{2}N$$
```python
    model.elasticity.s.half_satured() # Half satured assumption 
    model.elasticity.s.fill_sub_elasticity() # sum of 3 matrices that follow beta distributions
``` 

## Access to the variables
```python
    model.covariance
    model.rho
    model.MI
``` 