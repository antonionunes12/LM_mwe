# Levenberg-Marquardt minimum working examples

In this repository you will find minimal working examples of the Levenberg-Marquardt algorithm for the purposes of transitioning trajectories from the Circular Restricted Three-Body Problem (CR3BP) into a high-fidelity ephemeris model (HFEM), considering the attraction forces of the Earth, Moon, and Sun.

## Running

These examples are established in the form of Jupyter Notebooks and may be run in two ways.

To run them without the need for any installation steps, you can follow [this binder link](https://mybinder.org/v2/gh/antonionunes12/LM_mwe/master), which will open a browser-based environment that automatically installs all necessary dependencies on the cloud. Then, select one of the notebooks and proceed normally. Note that the first time the environment is built may take a considerable amout of time, until all dependencies have been installed.

Note also that some browsers have been found to have problems in establishing a connection to the python kernel -- the use of Mozilla Firefox or Google Chrome is recommended. In addition, beware that the limited computational power available in-cloud will make the examples provided slow to run at particular points along the code. 

If you want to run the examples locally, which will significantly improve run speed, you will need to first create the `tudat-space` environment to install `tudatpy` and its required dependencies, as described [here](https://docs.tudat.space/en/latest/getting-started/installation.html). This requires a `conda` installation. In addition, you will need to install Jupyter, if you haven't done yet:
```
conda install jupyter
````
To start, activate the tudat-space conda environment:
```
conda activate tudat-space
```
Then add the `tudat-space` environment to Jupyter:
```
python -m ipykernel install --user --name=tudat-space
```
Finally, you can create a local jupyter notebook instance:
```
jupyter notebook
```

Alternatively, if you use an IDE such as VS Code or other similar platforms, it may be possible to run the notebooks directly at the program's interface.

## Content

The examples provided are the following:
* **L2_Halo_QPO** : Showcase of the algorithm to determine a quasi-periodic counterpart to an L2 Halo orbit in the CR3BP over 10 revolutions. Only the base Levenberg-Marquardt is employed. *This is the ideal starting example which should be followed first*.
* **L2_L1_Transfer** : Showcase of the algorithm to determine a transfer trajectory from an L2 Halo orbit to an L1 Lyapunov orbit from the CR3BP. The full Levenberg-Marquardt algorithm, with the possibility for adaptive weighting is employed. *This is a more advanced example that requires the user to be acquainted with the first example*.