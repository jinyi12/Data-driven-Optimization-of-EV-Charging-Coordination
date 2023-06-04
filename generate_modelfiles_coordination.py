import gurobipy as gb
from gurobipy import GRB

import pandas as pd
import numpy as np
import os
import sys
import time
import datetime
import math
import pickle
import argparse
from tqdm import tqdm

# default configs in toml file
default_config = {
    "input_file": "Data/scenarios/1000_scenario_samples.pkl",
    "input_file_single": "Data/scenarios/scenarios_3.csv",
    "solve_batch": True,
}


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Parse args for solver')
    argparser.add_argument('--input_file', type=str, default=default_config.input_file, help='input file')
    argparser.add_argument('--input_file_single', type=str, default=default_config.input_file_single, help='input file single')
    argparser.add_argument('--solve_batch', type=bool, default=default_config.solve_batch, help='solve batch')
    
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

if __name__ == "__main__":

    # load "Data/scenarios/1000_scenario_samples.pkl" into all_scenarios
    with open(default_config["input_file"], "rb") as f:
        all_scenarios = pickle.load(f)
    
    # assert all_scenarios is a dictionary
    assert isinstance(all_scenarios, dict)
    
    # all_scenarios is a dictionary with keys as "scenarios" and "probabilities"
    # for each key, the value is a list of 1000 elements
    # each element is a numpy array of shape (48, 1)
    
    #print progress using tqdm 
    for n_scenario in tqdm(range(len(all_scenarios["scenarios"]))):  

        # read the input file
        # DayAheadSolarOutput = pd.read_csv(input_file, header=None).to_numpy()
        
        DayAheadSolarOutput = np.array(all_scenarios["scenarios"][n_scenario]).T
        print(DayAheadSolarOutput.shape)

        mean_in = 8
        std_in = 1
        mean_out = 16
        std_out = 1

        nbScenarios = 10


        model: gb.Model = gb.Model("coordination")

        # In the first stage simulation, rparking is equal to $1.5/per hour, and the
        # fixed daily operational cost is equal to $4000 per 24 h. The number of
        # charging stations in this EV parking deck is assumed to be 300.

        rparking = 1.5
        fixedcost = 4000
        nbStations = 100

        # nbTime = 48
        # timeInterval = 0.5
        
        nbTime = 24
        timeInterval = 1

        Gamma = 100000
        Epsilon = 0.00001

        # maximum charging power of each EV, in kW, using level 2 charging
        maxChargingPower = 19

        rebaterate = 0.6

        rcharging = 1.5

        # Battery capacity for $i$th EV in kWh
        BatteryCapacity = np.zeros(nbStations)

        # assume that the battery capacity of each EV is 64 kWh
        for i in range(nbStations):
            BatteryCapacity[i] = 64

        TOUPurchasePrice = np.zeros(nbTime)
        TOUSellPrice = np.zeros(nbTime)

        # TOU purchase price is 0.584/kWh on peak, 0.357 on mid-peak, and 0.281 on
        # off peak
        # TOU sell price is 10% cheaper than TOU purchase price
        # 08:00 to 11:00 mid-peak, 11:00 to 12:00 peak, 12:00 to 14:00 mid-peak,
        # 14:00 to 17:00 peak, 17:00 to 22:00 mid peak, 22:00 to 08:00 off-peak
        # 48 time intervals, each time interval is 30 minutes

        off_peak = 0.281
        mid_peak = 0.357
        peak = 0.584

        for t in range(nbTime):
            if t >= 0 and t < 16:
                TOUPurchasePrice[t] = off_peak
                TOUSellPrice[t] = 0.9 * off_peak
            if t >= 16 and t < 22:
                TOUPurchasePrice[t] = mid_peak
                TOUSellPrice[t] = 0.9 * mid_peak
            if t >= 22 and t < 24:
                TOUPurchasePrice[t] = peak
                TOUSellPrice[t] = 0.9 * peak
            if t >= 24 and t < 28:
                TOUPurchasePrice[t] = mid_peak
                TOUSellPrice[t] = 0.9 * mid_peak
            if t >= 28 and t < 34:
                TOUPurchasePrice[t] = peak
                TOUSellPrice[t] = 0.9 * peak
            if t >= 34 and t < 40:
                TOUPurchasePrice[t] = mid_peak
                TOUSellPrice[t] = 0.9 * mid_peak
            if t >= 40 and t < 48:
                TOUPurchasePrice[t] = off_peak
                TOUSellPrice[t] = 0.9 * off_peak

        #   ---------------------------------------------------------------------------
        #   *************** 1. Initialize variables for objective function
        #   ***************
        #   ---------------------------------------------------------------------------

        #   initialize predicted time out and time in variables, for each charging
        #   station
        tHout = np.zeros(nbStations)
        tHin = np.zeros(nbStations)

        tHout_vec = np.zeros(nbStations)
        tHin_vec = np.zeros(nbStations)

        #   read arrival and departure .csv files located in the data folder and
        #   initialize tHout and tHin, tPredicterIn and tPredictedOut
        current_path = os.getcwd()
        # print("Current path is : ", current_path)

        # extract the digit from file name where the file name is of the form "scenario_1.csv"
        # fileNumber = input_file.split("_")[1].split(".")[0]

        #   if the current path is not the root path, then go to parent path
        # if doesnt match certain keywords like DayAhead or Data-driven, then go to parent path
        if current_path.split("/")[-1].find("DayAheadForecast") == -1 and current_path.split("/")[-1].find("Data-driven") == -1:
            current_path = current_path.split("/")[:-1]
            current_path = "/".join(current_path)

        # print("Current path now is : ", current_path)

        arrival_file = "Data/arrival_departure_times.csv"
        arrival_path = os.path.join(current_path, arrival_file)

        #   read arrival and departure times from .csv file
        arrival_departure_vec_2D = np.genfromtxt(arrival_path, delimiter=",", skip_header=1)

        #   first column is arrival time, second column is departure time
        for i in range(nbStations):
            tHout_vec[i] = arrival_departure_vec_2D[i][1]
            tHout[i] = round(tHout_vec[i] / timeInterval)

            tHin_vec[i] = arrival_departure_vec_2D[i][0]
            tHin[i] = round(tHin_vec[i] / timeInterval)

        # convert tHout and tHin to integer
        tHout = tHout.astype(int)
        tHin = tHin.astype(int)

        #   read SOC distribution from .csv
        SOC_file = "Data/SOC.csv"
        SOC_path = os.path.join(current_path, SOC_file)

        SOC_vec = pd.read_csv(SOC_path, header=None).to_numpy()

        # read probability distribution from .csv
        # prob = pd.read_csv(current_path + "/Data/probabilities/probabilities_" + fileNumber + ".csv", header=None).to_numpy()
        
        prob = np.array(all_scenarios["probabilities"][n_scenario])

        #   initialize ON/OFF for predicted charging status, for each scenario,
        #   charging station, and time period
        #   initialize ON/OFF for charging status, for each scenario, charging station,
        #   and time period


        DayAheadOnOffChargingStatus = model.addVars(
            nbScenarios,
            nbStations,
            nbTime,
            vtype=gb.GRB.BINARY,
            name="DayAheadOnOffChargingStatus",
        )


        DayAheadChargingPower = model.addVars(
            nbScenarios,
            nbStations,
            nbTime,
            lb=0,
            ub=maxChargingPower,
            vtype=gb.GRB.CONTINUOUS,
            name="DayAheadChargingPower",
        )


        DayAheadUtilityPowerOutput = model.addVars(
            nbScenarios,
            nbTime,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=gb.GRB.CONTINUOUS,
            name="DayAheadUtilityPowerOutput",
        )


        DayAheadBuySellStatus = model.addVars(
            nbScenarios, nbTime, vtype=gb.GRB.BINARY, name="DayAheadBuySellStatus"
        )

        #   ---------------------------------------------------------------------------

        #   ---------------------------------------------------------------------------
        #   *************** 2. Initialize variables for constraints ***************
        #   ---------------------------------------------------------------------------

        #   initialize state-of-charge (SOC) decision variable at $t$th interval, under scenario $s$, for
        #   $i$th EV

        SOC = model.addVars(
            nbScenarios, nbStations, nbTime, lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name="SOC"
        )

        #   ---------------------------------------------------------------------------
        #   *************** 3. constraints ***************
        #   ---------------------------------------------------------------------------

        # 1. Power balance
        # Total power output from solar and utility grid == Sum of EV charging load

        for s in range(nbScenarios):
            for t in range(nbTime):
                model.addConstr(
                    DayAheadSolarOutput[s, t]
                    + DayAheadUtilityPowerOutput[s, t]
                    - DayAheadChargingPower.sum(s, "*", t)
                    == 0,
                    name="PowerBalance",
                )

        model.addConstrs(
            (
                DayAheadSolarOutput[s, t] * (DayAheadBuySellStatus[s, t] - 1)
                <= DayAheadUtilityPowerOutput[s, t]
                for s in range(nbScenarios)
                for t in range(nbTime)
            ),
            name="UtilityGridPowerOutput1",
        )
        model.addConstrs(
            (
                DayAheadUtilityPowerOutput[s, t] <= Gamma * DayAheadBuySellStatus[s, t]
                for s in range(nbScenarios)
                for t in range(nbTime)
            ),
            name="UtilityGridPowerOutput2",
        )

        # initial SOC

        times = np.arange(nbTime).astype(int)

        tHin_initial_indices_list = []
        t_between_tHin_tHout_indices_list = []
        t_greater_than_tHout_indices_list = []

        for i in range(nbStations):

            t_between_tHin_tHout_indices_list.append(
                np.where((times > tHin[i]) & (times <= tHout[i]))[0]
            )

            t_greater_than_tHout_indices_list.append(np.where(times > tHout[i])[0])

            tHin_initial_indices_list.append(np.where(times <= tHin[i])[0])

        # time the soc constraint addition time
        start = time.time()

        model.addConstrs(
            (
                SOC[s, i, t] == SOC_vec[i]
                for s in range(nbScenarios)
                for i in range(nbStations)
                for t in tHin_initial_indices_list[i]
            ),
            name="InitialSOC",
        )

        model.addConstrs(
            (
                SOC[s, i, t]
                == SOC[s, i, t - 1]
                + DayAheadChargingPower[s, i, t] * timeInterval / BatteryCapacity[i]
                for s in range(nbScenarios)
                for i in range(nbStations)
                for t in t_between_tHin_tHout_indices_list[i]
            ),
            name="BetweenSOC",
        )

        model.addConstrs(
            (
                SOC[s, i, t] == SOC[s, i, tHout[i]]
                for s in range(nbScenarios)
                for i in range(nbStations)
                for t in t_greater_than_tHout_indices_list[i]
            ),
            name="GreaterSOC",
        )

        model.addConstrs(
            (
                (
                    SOC[s, i, tHin[i]]
                    + DayAheadChargingPower.sum(s, i, "*")
                    * timeInterval
                    / BatteryCapacity[i]
                )
                == 1
                for s in range(nbScenarios)
                for i in range(nbStations)
            ),
            name="SOC1",
        )

        end = time.time()
        print("Time to add SOC constraints: " + str(end - start))

        # 4. Charging power limit (part 2)
        # Second and third constraint ensures that the charging power for every
        # scenario lies between the range $(0, P_{\text{level}}]$ when
        # $\hat{u}_{s, i}(t) == 1$. $\hat{P}_{s, i}(t) == 0 $ if $\hat{u}_{s,
        # i}(t) == 0$.

        model.addConstrs(
            (
                DayAheadOnOffChargingStatus[s, i, t] == 0
                for s in range(nbScenarios)
                for i in range(nbStations)
                for t in range(nbTime)
                if t < tHin[i] or t > tHout[i]
            ),
            name="OnOffChargingStatus",
        )
        model.addConstrs(
            (
                DayAheadChargingPower[s, i, t]
                <= DayAheadOnOffChargingStatus[s, i, t] * maxChargingPower
                for s in range(nbScenarios)
                for i in range(nbStations)
                for t in range(nbTime)
            ),
            name="ChargingPower1",
        )
        model.addConstrs(
            (
                DayAheadChargingPower[s, i, t]
                >= Epsilon * DayAheadOnOffChargingStatus[s, i, t]
                for s in range(nbScenarios)
                for i in range(nbStations)
                for t in range(nbTime)
            ),
            name="ChargingPower2",
        )

        # time to make charging power linear

        # start = time.time()

        DayAheadChargingPowerLinearized = model.addVars(
            nbScenarios,
            nbStations,
            nbTime,
            lb=0,
            ub=maxChargingPower,
            vtype=GRB.CONTINUOUS,
            name="DayAheadChargingPowerLinearized",
        )

        for s in range(nbScenarios):
            for i in range(nbStations):
                for t in range(nbTime):

                    DayAheadChargingPowerLinearized_s_i_t = DayAheadChargingPowerLinearized[
                        s, i, t
                    ]

                    DayAheadOnOffChargingStatus_s_i_t = DayAheadOnOffChargingStatus[s, i, t]

                    DayAheadChargingPower_s_i_t = DayAheadChargingPower[s, i, t]

                    model.addConstr(
                        DayAheadChargingPowerLinearized_s_i_t
                        <= maxChargingPower * DayAheadOnOffChargingStatus_s_i_t
                    )
                    model.addConstr(
                        DayAheadChargingPowerLinearized_s_i_t <= DayAheadChargingPower_s_i_t
                    )
                    model.addConstr(DayAheadChargingPowerLinearized_s_i_t >= 0)
                    model.addConstr(
                        DayAheadChargingPowerLinearized_s_i_t
                        >= DayAheadChargingPower_s_i_t
                        - (1 - DayAheadOnOffChargingStatus_s_i_t) * maxChargingPower
                    )

        # end = time.time()
        # print("Time to add charging power linearization constraints: " + str(end - start))

        # time to make linearized output power purchase
        start = time.time()

        DayAheadUtilityPowerOutputLinearizedPurchase = model.addVars(
            nbScenarios,
            nbTime,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="DayAheadUtilityPowerOutputLinearizedPurchase",
        )

        for s in range(nbScenarios):
            for t in range(nbTime):

                DayAheadUtilityPowerOutputLinearizedPurchase_s_t = (
                    DayAheadUtilityPowerOutputLinearizedPurchase[s, t]
                )

                DayAheadBuySellStatus_s_t = DayAheadBuySellStatus[s, t]

                model.addConstr(
                    DayAheadUtilityPowerOutputLinearizedPurchase_s_t
                    <= Gamma * DayAheadBuySellStatus_s_t
                )
                model.addConstr(
                    DayAheadUtilityPowerOutputLinearizedPurchase_s_t
                    >= -Gamma * DayAheadBuySellStatus_s_t
                )
                model.addConstr(DayAheadUtilityPowerOutputLinearizedPurchase_s_t <= Gamma)
                model.addConstr(DayAheadUtilityPowerOutputLinearizedPurchase_s_t >= -Gamma)
                model.addConstr(
                    DayAheadUtilityPowerOutputLinearizedPurchase_s_t
                    >= DayAheadUtilityPowerOutput[s, t]
                    - (1 - DayAheadBuySellStatus_s_t) * Gamma
                )
                model.addConstr(
                    DayAheadUtilityPowerOutputLinearizedPurchase_s_t
                    <= DayAheadUtilityPowerOutput[s, t]
                    - (1 - DayAheadBuySellStatus_s_t) * -Gamma
                )
                model.addConstr(
                    DayAheadUtilityPowerOutputLinearizedPurchase_s_t
                    <= DayAheadUtilityPowerOutput[s, t]
                    + (1 - DayAheadBuySellStatus_s_t) * Gamma
                )

        # end = time.time()
        # print("Time to make linearized output power purchase: " + str(end - start))

        #   ---------------------------------------------------------------------------

        #   ---------------------------------------------------------------------------
        #   *************** 4. objective function ***************
        #   ---------------------------------------------------------------------------

        # The sum of the differences between the time of EV exit
        # ($\hat{T}_{\text{out}, i}$) and the time of EV entry
        # ($\hat{T}_{\text{in}, i}$) for all $i$ from 1 to $N$, and then subtract
        # parking fee rebate rate * time interval * prob[s] *
        # DayAheadOnOffChargingStatus .

        expr_tHatParking = gb.LinExpr()
        for s in range(nbScenarios):
            for i in range(nbStations):
                expr_tHatParking.add(tHout[i])
                expr_tHatParking.add(-tHin[i])
                for t in range(nbTime):
                    expr_tHatParking.add(
                        DayAheadOnOffChargingStatus[s, i, t],
                        -(rebaterate * timeInterval * prob[s]),
                    )

        # define expression for \hat{R}_{\text {parking }}=r_{\text {parking }}
        # \sum_{i=1}^N \hat{t}_{\text {parking }, i}

        expr_RHatParking = rparking * expr_tHatParking

        # time to build expression
        start = time.time()
        expr_RHatCharging = gb.LinExpr()
        for s in range(nbScenarios):
            for t in range(nbTime):

                expr_RHatCharging.add(
                    DayAheadChargingPowerLinearized.sum(s, "*", t),
                    prob[s] * (rcharging * timeInterval),
                )

                expr_RHatCharging.add(
                    DayAheadUtilityPowerOutputLinearizedPurchase[s, t],
                    -prob[s] * (TOUPurchasePrice[t] * timeInterval),
                )

                expr_RHatCharging.add(
                    DayAheadUtilityPowerOutput[s, t],
                    -prob[s] * (TOUSellPrice[t] * timeInterval),
                )
                expr_RHatCharging.add(
                    -DayAheadUtilityPowerOutputLinearizedPurchase[s, t],
                    -prob[s] * (TOUSellPrice[t] * timeInterval),
                )

        # end = time.time()
        # print("time to build expression", end - start)

        #  expr_RHat = expr_RHatParking + expr_RHatCharging - fixedcost;
        model.setObjective(expr_RHatParking + expr_RHatCharging - fixedcost, GRB.MAXIMIZE)

        model.update()

        # save model to .lp file at "instances/mip/data/coordination/"
        model.write("instances/mip/data/coordination/coordination_{}.lp".format(n_scenario))
        
        
