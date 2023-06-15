# Data-driven-Optimization-of-EV-Charging-Coordination

This project explores data-driven optimization for EV charging coordination at a parking deck. The optimization problem is formulated as a stochastic optimization of EV charging coordination at a parking deck, with varying solar generation scenarios. Data utilized in this project include Malaysia's solar irradiance data for the year 2020. All other data are generated and simulated.

## To generate solar generation scenarios and their respective scenario probabilities
Run 
```{python}
python generate_solar_scenarios.py
```
The script takes in a command line argument for the number of scenarios to be generated. This will generate `nSamples` sets of solar generation scenarios, and their respective scenario probabilities. By default `nScenarios=100` and each set of solar generation scenarios is obtained via scenario reduction from an initial amount of 100 solar scenarios to `n_sc_red=10` scenarios. Please modify these parameters according to your own needs. 

The script outputs a `.pkl` file at `f"Data/scenarios/{nSamples}_scenario_samples.pkl"` that contains `nSamples` unique sets of pair of solar generation scenarios and scenario probabilities.

## Generating initial State-of-Charge (SOC) data
Run 
```{python}
python SOC.py
```
The script generates a normal distribution of initial SOC values to be used for the formulated optimization problem. 
Default values are:
```{python}
nbStations = 300
mean = 0.6
std = 0.1
```
The output data will be located at `"Data/SOC.csv"`

## Building optimization models
Run 
```{python}
python generate_modelfiles_coordination.py
```

