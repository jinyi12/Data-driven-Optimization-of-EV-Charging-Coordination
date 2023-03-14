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


if __name__ == "__main__":

    # if not enough arguments, use the default input file
    if len(sys.argv) < 2:
        print("No input file specified, using default input file")
        # input_file = "/Users/jinyiyong/Documents/Optimization/DayAheadForecast/Data/scenarios/scenarios_2.csv"
        input_file = "Data/scenarios/scenarios_3.csv"
    else:
        input_file = sys.argv[1]

    # read the input file
    DayAheadSolarOutput = pd.read_csv(input_file, header=None).to_numpy()

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
    nbStations = 300

    nbTime = 48
    timeInterval = 0.5

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
    print("Current path is : ", current_path)

    # extract the digit from file name where the file name is of the form "scenario_1.csv"
    fileNumber = input_file.split("_")[1].split(".")[0]

    #   if the current path is not the root path, then go to parent path
    # if doesnt match certain keywords like DayAhead or Data-driven, then go to parent path
    if current_path.split("/")[-1].find("DayAheadForecast") == -1 and current_path.split("/")[-1].find("Data-driven") == -1:
        current_path = current_path.split("/")[:-1]
        current_path = "/".join(current_path)

    print("Current path now is : ", current_path)

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
    prob = pd.read_csv(current_path + "/Data/probabilities/probabilities_" + fileNumber + ".csv", header=None).to_numpy()

    #   initialize ON/OFF for predicted charging status, for each scenario,
    #   charging station, and time period
    #   initialize ON/OFF for charging status, for each scenario, charging station,
    #   and time period

    # DayAheadOnOffChargingStatus = model.addMVar(
    #     shape=(nbScenarios, nbStations, nbTime),
    #     vtype=gb.GRB.BINARY,
    #     name="DayAheadOnOffChargingStatus",
    # )

    DayAheadOnOffChargingStatus = model.addVars(
        nbScenarios,
        nbStations,
        nbTime,
        vtype=gb.GRB.BINARY,
        name="DayAheadOnOffChargingStatus",
    )

    # DayAheadChargingPower = model.addMVar(
    #     shape=(nbScenarios, nbStations, nbTime),
    #     lb=0,
    #     ub=maxChargingPower,
    #     vtype=gb.GRB.CONTINUOUS,
    #     name="DayAheadChargingPower",
    # )

    DayAheadChargingPower = model.addVars(
        nbScenarios,
        nbStations,
        nbTime,
        lb=0,
        ub=maxChargingPower,
        vtype=gb.GRB.CONTINUOUS,
        name="DayAheadChargingPower",
    )

    # DayAheadUtilityPowerOutput = model.addMVar(
    #     shape=(nbScenarios, nbTime),
    #     lb=-GRB.INFINITY,
    #     ub=GRB.INFINITY,
    #     vtype=gb.GRB.CONTINUOUS,
    #     name="DayAheadUtilityPowerOutput",
    # )

    DayAheadUtilityPowerOutput = model.addVars(
        nbScenarios,
        nbTime,
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        vtype=gb.GRB.CONTINUOUS,
        name="DayAheadUtilityPowerOutput",
    )

    # DayAheadBuySellStatus = model.addMVar(
    #     shape=(nbScenarios, nbTime), vtype=gb.GRB.BINARY, name="DayAheadBuySellStatus"
    # )

    DayAheadBuySellStatus = model.addVars(
        nbScenarios, nbTime, vtype=gb.GRB.BINARY, name="DayAheadBuySellStatus"
    )

    #   ---------------------------------------------------------------------------

    #   ---------------------------------------------------------------------------
    #   *************** 2. Initialize variables for constraints ***************
    #   ---------------------------------------------------------------------------

    #   initialize state-of-charge (SOC) decision variable at $t$th interval, under scenario $s$, for
    #   $i$th EV

    # SOC = model.addMVar(
    #     shape=(nbScenarios, nbStations, nbTime),
    #     lb=0,
    #     ub=1,
    #     vtype=gb.GRB.CONTINUOUS,
    #     name="SOC",
    # )

    SOC = model.addVars(
        nbScenarios, nbStations, nbTime, lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name="SOC"
    )

    #   ---------------------------------------------------------------------------
    #   *************** 3. constraints ***************
    #   ---------------------------------------------------------------------------

    # 1. Power balance
    # Total power output from solar and utility grid == Sum of EV charging load

    #   ExprArray2D PowerBalance(env, nbScenarios);
    #   for (int s = 0; s < nbScenarios; s++) {
    #     PowerBalance[s] = IloExprArray(env, nbTime);
    #     for (int t = 0; t < nbTime; t++) {
    #       PowerBalance[s][t] = IloExpr(env);
    #       PowerBalance[s][t] += DayAheadSolarOutput[s][t];
    #       PowerBalance[s][t] += DayAheadUtilityPowerOutput[s][t];
    #       for (int i = 0; i < nbStations; i++) {
    #         PowerBalance[s][t] -= DayAheadChargingPower[s][i][t];
    #       }
    #       model.add(PowerBalance[s][t] == 0);
    #     }
    #   }

    # PowerBalance = DayAheadSolarOutput + DayAheadUtilityPowerOutput
    # for i in range(nbStations):
    #     PowerBalance -= DayAheadChargingPower[:, i, :]
    # model.addConstrs(
    #     (PowerBalance[s, t] == 0 for s in range(nbScenarios) for t in range(nbTime)),
    #     name="PowerBalance",
    # )

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

    # for s in range(nbScenarios):
    #     # time the loop
    #     start = time.time()
    #     for i in range(nbStations):
    #         expr = SOC[s, i, tHin[i]]
    #         for t in range(nbTime):
    #             # if t <= tHin[i]:
    #             #     model.addConstr(SOC[s, i, t] == SOC_vec[i])
    #             # if t > tHin[i] and t <= tHout[i]:
    #             #     model.addConstr(
    #             #         SOC[s, i, t]
    #             #         == SOC[s, i, t - 1]
    #             #         + DayAheadChargingPower[s, i, t]
    #             #         * timeInterval
    #             #         / BatteryCapacity[i]
    #             #     )
    #             # if t > tHout[i]:
    #             #     model.addConstr(SOC[s, i, t] == SOC[s, i, tHout[i]])

    #             expr += (
    #                 DayAheadChargingPower[s, i, t] * timeInterval / BatteryCapacity[i]
    #             )

    #         model.addConstr(expr == 1)

    #     end = time.time()
    #     print(
    #         "Time to add SOC constraints for scenario "
    #         + str(s)
    #         + ": "
    #         + str(end - start)
    #     )

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

    # Linearize DayAheadChargingPower * DayAheadOnOffChargingStatus
    # DayAheadChargingPowerLinearized = model.addMVar(
    #     nbScenarios,
    #     nbStations,
    #     nbTime,
    #     lb=0,
    #     ub=maxChargingPower,
    #     vtype=GRB.CONTINUOUS,
    #     name="DayAheadChargingPowerLinearized",
    # )

    # time to make charging power linear

    start = time.time()

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

    end = time.time()
    print("Time to add charging power linearization constraints: " + str(end - start))

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

    end = time.time()
    print("Time to make linearized output power purchase: " + str(end - start))

    #   ---------------------------------------------------------------------------

    #   ---------------------------------------------------------------------------
    #   *************** 4. objective function ***************
    #   ---------------------------------------------------------------------------

    # The sum of the differences between the time of EV exit
    # ($\hat{T}_{\text{out}, i}$) and the time of EV entry
    # ($\hat{T}_{\text{in}, i}$) for all $i$ from 1 to $N$, and then subtract
    # parking fee rebate rate * time interval * prob[s] *
    # DayAheadOnOffChargingStatus .

    #       IloExpr expr_tHatParking(env);
    #   for (int s = 0; s < nbScenarios; s++) {
    #     for (int i = 0; i < nbStations; i++) {
    #       expr_tHatParking += (tHout[i] - tHin[i]);
    #       for (int t = 0; t < nbTime; t++) {
    #         expr_tHatParking -= (rebaterate * timeInterval * prob[s] *
    #                              DayAheadOnOffChargingStatus[s][i][t]);
    #       }
    #     }
    #   }
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

    # Convert the following CPLEX code to Gurobi
    """
    IloExpr expr_RHatCharging(env);
    for (int s = 0; s < nbScenarios; s++) {
        for (int t = 0; t < nbTime; t++) {
        for (int i = 0; i < nbStations; i++) {
            expr_RHatCharging +=
                prob[s] * (rcharging * DayAheadChargingPowerLinearized[s][i][t] *
                        timeInterval);
        }

        expr_RHatCharging -=
            prob[s] * (DayAheadUtilityPowerOutputLinearizedPurchase[s][t] *
                        TOUPurchasePrice[t] * timeInterval);
        expr_RHatCharging -=
            prob[s] * ((DayAheadUtilityPowerOutput[s][t] -
                        DayAheadUtilityPowerOutputLinearizedPurchase[s][t]) *
                        TOUSellPrice[t] * timeInterval);
        }
    }

    """

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

    end = time.time()
    print("time to build expression", end - start)

    #  expr_RHat = expr_RHatParking + expr_RHatCharging - fixedcost;
    model.setObjective(expr_RHatParking + expr_RHatCharging - fixedcost, GRB.MAXIMIZE)

    model.update()

    # mp = model.presolve()

    # get constraint matrix
    A = model.getA()

    # get the right hand side
    rhs = model.getAttr("RHS", model.getConstrs())
    rhs = np.array(rhs)

    # get the coefficients of the objective function
    cost_vectors = model.getAttr("Obj", model.getVars())
    cost_vectors = np.array(cost_vectors)

    # get output
    model.optimize()

    # get the vars by name
    DayAheadBuySellStatusVars = [var for var in model.getVars() if "DayAheadBuySellStatus" in var.varName]
    DayAheadOnOffChargingStatusVars = [var for var in model.getVars() if "DayAheadOnOffChargingStatus" in var.varName]
    DayAheadChargingPowerVars = [var for var in model.getVars() if "DayAheadChargingPower" in var.varName]
    DayAheadUtilityPowerOutputVars = [var for var in model.getVars() if "DayAheadUtilityPowerOutput" in var.varName]
    SOCVars = [var for var in model.getVars() if "SOC" in var.varName]

    output_vars = [DayAheadBuySellStatusVars, DayAheadOnOffChargingStatusVars, DayAheadChargingPowerVars, DayAheadUtilityPowerOutputVars]

    # get the solution for each variable above
    DayAheadBuySellStatusSolution = model.getAttr("X", DayAheadBuySellStatusVars)
    DayAheadOnOffChargingStatusSolution = model.getAttr("X", DayAheadOnOffChargingStatusVars)
    DayAheadChargingPowerSolution = model.getAttr("X", DayAheadChargingPowerVars)
    DayAheadUtilityPowerOutputSolution = model.getAttr("X", DayAheadUtilityPowerOutputVars)
    SOCSolution = model.getAttr("X", SOCVars)


    # concatenate the solution without SOC
    output = np.concatenate((DayAheadBuySellStatusSolution, DayAheadOnOffChargingStatusSolution))

    solution_dict = dict()
    solution_dict["DayAheadBuySellStatus"] = DayAheadBuySellStatusSolution
    solution_dict["DayAheadOnOffChargingStatus"] = DayAheadOnOffChargingStatusSolution
    solution_dict["DayAheadChargingPower"] = DayAheadChargingPowerSolution
    solution_dict["DayAheadUtilityPowerOutput"] = DayAheadUtilityPowerOutputSolution
    solution_dict["SOC"] = SOCSolution
    solution_dict["output"] = output

    # print the shape of the solution
    print("target shape", output.shape)

    # write solution_dict to a file
    with open(current_path + "/Data/output/solution_dict_" + fileNumber + ".pkl", "wb") as f:
        pickle.dump(solution_dict, f)
    
    # input dictionary
    input_dict = dict()
    input_dict["A"] = A
    input_dict["b"] = rhs
    input_dict["c"] = cost_vectors

    print("A shape", A.shape)
    print("b shape", rhs.shape)
    print("c shape", cost_vectors.shape)

    # write input_dict to a file
    with open(current_path + "/Data/output/input_dict_" + fileNumber + ".pkl", "wb") as f:
        pickle.dump(input_dict, f)
    
