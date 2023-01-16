#include "Function.h"
#include "ilconcert/iloenv.h"
#include "ilconcert/iloexpression.h"
#include "ilconcert/ilosys.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ilconcert/ilocsvreader.h>
#include <ilcplex/ilocplex.h>
#include <iostream>
#include <random>
#include <string>
#include <time.h>

#include <cmath>
#include <stdlib.h>
#include <vector>

ILOSTLBEGIN

typedef IloArray<IloNumVarArray> NumVarMatrix2D;
typedef IloArray<NumVarMatrix2D> NumVarMatrix3D;
typedef IloArray<IloNumArray> NumMatrix2D;
typedef IloArray<NumMatrix2D> NumMatrix3D;
typedef IloArray<IloExprArray> ExprArray2D;

// void matread(const char *file, const char *name, std::vector<double> &v);

int main(int, char **) {
    printf("Hello World!");

    std::mt19937 rng;

    double mean_in = 8;
    double std_in = 1;
    double mean_out = 16;
    double std_out = 1;

    std::normal_distribution<double> norm_dist_in(mean_in, std_in);
    std::normal_distribution<double> norm_dist_out(mean_out, std_out);
    std::normal_distribution<double> norm_dist_SOC(0.6, 0.1);

    int nbScenarios = 10;

    IloEnv env;
    IloModel model(env);
    IloCplex cplex(env);

    // In the first stage simulation, rparking is equal to $1.5/per hour, and the
    // fixed daily operational cost is equal to $4000 per 24 h. The number of
    // charging stations in this EV parking deck is assumed to be 300.

    // start of real time optimization window
    IloInt t0 = 0;

    // real time horizon
    IloInt H = 4;

    IloNum SOC_max = 1.0;
    IloNum rparking = 1.5;
    IloNum fixedcost = 4000;
    IloInt nbStations = 300;

    IloNum nbTime = 48;
    IloNum timeInterval = 0.5;

    // IloNum rebateRate = 0.5;
    IloNum Gamma = 10000;
    IloNum Epsilon = 0.0001;

    // maximum charging power of each EV, in kW, using level 2 charging
    IloNum maxChargingPower = 19;

    IloNumVar rebaterate, rcharging;

    // Battery capacity for $i$th EV in kWh
    IloNumArray BatteryCapacity(env, nbStations);

    // assume that the battery capacity of each EV is 64 kWh
    for (int i = 0; i < nbStations; i++) {
        BatteryCapacity[i] = 64;
    }

    // probability of scenario s happening
    IloNumArray prob(env, nbScenarios);

    // probability of scenario s happening is 1/nbScenarios
    for (int s = 0; s < nbScenarios; s++) {
        prob[s] = 1.0 / nbScenarios;
    }

    // TOU purchase price
    IloNumArray TOUPurchasePrice(env, nbTime);
    IloNumArray TOUSellPrice(env, nbTime);

    // ---------------------------------------------------------------------------
    // *************** 1. Initialize variables for objective function
    // ***************
    // ---------------------------------------------------------------------------

    // initialize predicted time out and time in variables, for each charging
    // station
    IloNumArray tHout(env, nbStations, 0, nbTime, ILOINT);
    IloNumArray tHin(env, nbStations, 0, nbTime, ILOINT);
    IloNumArray tPredictedOut(env, nbStations, 0, nbTime, ILOINT);
    IloNumArray tPredictedIn(env, nbStations, 0, nbTime, ILOINT);

    std::vector<double> tout_vec(nbStations);
    std::vector<double> tin_vec(nbStations);
    std::vector<double> tPredictedIn_vec(nbStations);
    std::vector<double> tPredictedOut_vec(nbStations);

    // read arrival and departure .csv files located in the data folder and
    // initialize tHout and tHin, tPredicterIn and tPredictedOut
    std::filesystem::path current_path = std::filesystem::current_path();
    std::cout << "Current path is : " << current_path << std::endl;
    // go to parent path
    current_path = current_path.parent_path();

    std::cout << "Current path now is : " << current_path << std::endl;

    const char *arrival_file = "Data/arrival_distribution.csv";
    std::filesystem::path arrival_path = current_path / arrival_file;
    const char *arrival_path_char = arrival_path.c_str();

    const char *departure_file = "Data/departure_distribution.csv";
    std::filesystem::path departure_path = current_path / departure_file;
    const char *departure_path_char = departure_path.c_str();

    const char *arrival_predicted_file = "Data/arrival_distribution_predicted.csv";
    std::filesystem::path arrival_predicted_path = current_path / arrival_predicted_file;
    const char *arrival_predicted_path_char = arrival_predicted_path.c_str();

    const char *departure_predicted_file =
            "Data/departure_distribution_predicted.csv";
    std::filesystem::path departure_predicted_path = current_path / departure_predicted_file;
    const char *departure_predicted_path_char = departure_predicted_path.c_str();

    // read arrival distribution
    IloCsvReader arrival_reader(env, arrival_path_char);
    IloCsvReader::LineIterator arrival_it(arrival_reader);
    IloCsvLine arrival_line = *arrival_it;

    int index = 0;
    while (arrival_it.ok()) {
        tHin[index] = arrival_line.getFloatByPosition(0);
        ++arrival_it;
    }

    // read departure distribution
    std::ifstream file(departure_path_char);
    std::string line;
    if (file.is_open()) {
        while (getline(file, line)) {
            tout_vec.push_back(std::stod(line));
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }


    for (int i = 0; i < nbStations; i++) {
        tHout[i] = tout_vec[i];
        tHin[i] = tin_vec[i];
        tPredictedIn[i] = tPredictedIn_vec[i];
        tPredictedOut[i] = tPredictedOut_vec[i];
    }

    // initialize ON/OFF for predicted charging status, for each scenario,
    // charging station, and time period
    // initialize ON/OFF for charging status, for each scenario, charging station,
    // and time period
    NumVarMatrix3D DayAheadOnOffChargingStatus(env, nbScenarios);
    for (int s = 0; s < nbScenarios; s++) {
        DayAheadOnOffChargingStatus[s] = NumVarMatrix2D(env, nbStations);
        for (int i = 0; i < nbStations; i++) {
            DayAheadOnOffChargingStatus[s][i] =
                    IloNumVarArray(env, nbTime, 0, 1, ILOINT);
        }
    }

    // initialize Day-ahead charging power for $i$th EV, at $t$th interval, under
    // $s$th scenario, in kW
    NumVarMatrix3D DayAheadChargingPower(env, nbScenarios);
    for (int s = 0; s < nbScenarios; s++) {
        DayAheadChargingPower[s] = NumVarMatrix2D(env, nbStations);
        for (int i = 0; i < nbStations; i++) {
            DayAheadChargingPower[s][i] = IloNumVarArray(env, nbTime, 0, maxChargingPower, ILOFLOAT);
        }
    }

    // intialize Day-ahead power output from utility grid at $t$th interval, under
    // $s$th scenario, in kW
    NumVarMatrix2D DayAheadUtilityPowerOutput(env, nbScenarios);
    for (int s = 0; s < nbScenarios; s++) {
        DayAheadUtilityPowerOutput[s] = IloNumVarArray(env, nbTime, 0 , IloInfinity);
    }

    // intialize day-ahead buy/sell (1/0) status at $t$th interval, under $s$th
    // scenario
    NumVarMatrix2D DayAheadBuySellStatus(env, nbScenarios);
    for (int s = 0; s < nbScenarios; s++) {
        DayAheadBuySellStatus[s] = IloNumVarArray(env, nbTime, 0, 1, ILOINT);
    }

    // initialize electricity purchase price from the grid at $t$th time interval
    // ($/kWh)
    IloNumArray ElectricityPurchasePrice(env, nbTime);

    // ---------------------------------------------------------------------------

    // ---------------------------------------------------------------------------
    // *************** 2. Initialize variables for constraints ***************
    // ---------------------------------------------------------------------------

    // initialize Day-ahead solar output power in $t$th interval, under scenario
    // $s$ in kW
    NumVarMatrix2D DayAheadSolarOutput(env, nbScenarios);
    for (int s = 0; s < nbScenarios; s++) {
        DayAheadSolarOutput[s] = IloNumVarArray(env, nbTime, 0, IloInfinity);
    }

    // initialize state-of-charge (SOC) at $t$th interval, under scenario $s$, for
    // $i$th EV
    NumVarMatrix3D SOC(env, nbScenarios);
    for (int s = 0; s < nbScenarios; s++) {
        SOC[s] = NumVarMatrix2D(env, nbStations);
        for (int i = 0; i < nbStations; i++) {
            SOC[s][i] = IloNumVarArray(env, nbTime, 0, 1, ILOFLOAT);
        }
    }

    // ---------------------------------------------------------------------------
    // *************** 3. constraints ***************
    // ---------------------------------------------------------------------------

    // 1. Power balance
    // Total power output from solar and utility grid == Sum of EV charging load

    for (int s = 0; s < nbScenarios; s++) {
        for (int t = 0; t < nbTime; t++) {
            IloExpr expr(env);
            expr += DayAheadSolarOutput[s][t];
            expr += DayAheadUtilityPowerOutput[s][t];
            for (int i = 0; i < nbStations; i++) {
                expr -= DayAheadChargingPower[s][i][t];
            }
            model.add(expr == 0);
            expr.end();
        }
    }

    // 2. Utility grid power output

    for (int s = 0; s < nbScenarios; s++) {
        for (int t = 0; t < nbTime; t++) {
            model.add(DayAheadSolarOutput[s][t] * (DayAheadBuySellStatus[s][t] - 1) <=
                      DayAheadUtilityPowerOutput[s][t]);
            model.add(DayAheadUtilityPowerOutput[s][t] <=
                      Gamma * DayAheadBuySellStatus[s][t]);
        }
    }

    // 3. Charging power limit (part 1)
    // for each scenario, for each EV, SOC + total increasing SoC (which is
    // day-ahead charging power/Battery capacity) of $i$th EV across the whole
    // charging period == 1
    for (int s = 0; s < nbScenarios; s++) {
        for (int i = 0; i < nbStations; i++) {
            IloExpr expr(env);
            expr += SOC[s][i][tHin[i]];
            for (int t = 0; t < nbTime; t++) {
                expr += DayAheadChargingPower[s][i][t] / BatteryCapacity[i];
            }
            model.add(expr == 1);
            expr.end();
        }
    }

    // 4. Charging power limit (part 2)
    // Second and third constraint ensures that the charging power for every
    // scenario lies between the range $(0, P_{\text{level}}]$ when $\hat{u}_{s,
    // i}(t) == 1$. $\hat{P}_{s, i}(t) == 0 $ if $\hat{u}_{s, i}(t) == 0$.

    for (int s = 0; s < nbScenarios; s++) {
        for (int i = 0; i < nbStations; i++) {
            for (int t = 0; t < nbTime; t++) {
                model.add(DayAheadChargingPower[s][i][t] <=
                          DayAheadOnOffChargingStatus[s][i][t] * maxChargingPower);
                model.add(DayAheadChargingPower[s][i][t] >= 0);
            }
        }
    }

    // 5. Charging power limit (part 3)
    // \varepsilon \hat{u}_{s, i}(t) & \leq \hat{P}_{s, i}(t) & \leq P_{\text
    // {level }} \hat{u}_{s, i}(t) & \forall s, \forall i, \forall t
    for (int s = 0; s < nbScenarios; s++) {
        for (int i = 0; i < nbStations; i++) {
            for (int t = 0; t < nbTime; t++) {
                model.add(DayAheadChargingPower[s][i][t] <=
                          DayAheadOnOffChargingStatus[s][i][t] * maxChargingPower);
                model.add(DayAheadChargingPower[s][i][t] >=
                          Epsilon * DayAheadOnOffChargingStatus[s][i][t]);
            }
        }
    }

    // ---------------------------------------------------------------------------

    // ---------------------------------------------------------------------------
    // *************** 4. objective function ***************
    // ---------------------------------------------------------------------------

    // define expression for \begin{aligned}
    // & \sum_{i=1}^N \hat{t}_{\text {parking },
    // i}=\sum_{i=1}^N\left[\hat{T}_{\text {out }, i}-\hat{T}_{\mathrm{in},
    // i}\right] \\
  // & -\rho \Delta t \sum_{s=1}^S \operatorname{prob}_s \sum_{i=1}^N
    // \sum_{t=1}^T \hat{u}_{s, i}(t) \\
  // &
    // \end{aligned}

    // The sum of the differences between the time of EV exit
    // ($\hat{T}_{\text{out}, i}$) and the time of EV entry ($\hat{T}_{\text{in},
    // i}$) for all $i$ from 1 to $N$, and then subtract parking fee rebate rate *
    // time interval * prob[s] * DayAheadOnOffChargingStatus .
    IloExpr expr_tHatParking(env);
    for (int s = 0; s < nbScenarios; s++) {
        for (int i = 0; i < nbStations; i++) {
            for (int t = 0; t < nbTime; t++) {
                expr_tHatParking +=
                        (tHout[i] - tHin[i]) - (rebaterate * timeInterval * prob[s] *
                                                DayAheadOnOffChargingStatus[s][i][t]);
            }
        }
    }

    // define expression for \hat{R}_{\text {parking }}=r_{\text {parking }}
    // \sum_{i=1}^N \hat{t}_{\text {parking }, i}
    IloExpr expr_RHatParking(env);
    expr_RHatParking = rparking * expr_tHatParking;

    // define expression for expected day-ahead charging revenue
    // \hat{R}_{\text {charging }}=\sum_{s=1}^S
    // \operatorname{prob}_s \sum_{t=1}^T[ & r_{\text {charging }} \sum_{i=1}^N
    // \hat{P}_{s, i}(t) \Delta t -\hat{P}_{\text {grid }, s}(t) \hat{y}_s(t)
    // c(t) \Delta t - \hat{P}_{\text {grid }, s} (1-\hat{y}_s(t)) h(t) \Delta t

    IloExpr expr_RHatCharging(env);
    for (int s = 0; s < nbScenarios; s++) {
        for (int t = 0; t < nbTime; t++) {
            IloExpr expr_RHatCharging_i(env);
            for (int i = 0; i < nbStations; i++) {
                expr_RHatCharging_i +=
                        (rcharging * DayAheadChargingPower[s][i][t] * timeInterval -
                         DayAheadUtilityPowerOutput[s][t] * DayAheadBuySellStatus[s][t] *
                         TOUPurchasePrice[t] * timeInterval -
                         DayAheadUtilityPowerOutput[s][t] *
                         (1 - DayAheadBuySellStatus[s][t]) * TOUSellPrice[t] * timeInterval);
            }
            expr_RHatCharging += prob[s] * expr_RHatCharging_i;
        }
    }

    // define expression for \hat{R} = \hat{R}_{\text {charging }} + \hat{R}_{\text
    // {parking }} - C_{\text {operation }}
    IloExpr expr_RHat(env);
    expr_RHat = expr_RHatParking + expr_RHatCharging - fixedcost;

    model.add(IloMaximize(env, expr_RHat));
    expr_RHat.end();
    expr_RHatParking.end();
    expr_RHatCharging.end();
    expr_tHatParking.end();

    return 0;
}