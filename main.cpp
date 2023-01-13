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
  IloNumArray tout(env, nbStations, 0, nbTime, ILOINT);
  IloNumArray tin(env, nbStations, 0, nbTime, ILOINT);
  IloNumArray tPredictedOut(env, nbStations, 0, nbTime, ILOINT);
  IloNumArray tPredictedIn(env, nbStations, 0, nbTime, ILOINT);

  std::vector<double> tout_vec(nbStations);
  std::vector<double> tin_vec(nbStations);
  std::vector<double> tPredictedIn_vec(nbStations);
  std::vector<double> tPredictedOut_vec(nbStations);

  // read arrival and departure .csv files located in the data folder and
  // initialize tout and tin, tPredicterIn and tPredictedOut
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
    tin[index] = arrival_line.getFloatByPosition(0);
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
    tout[i] = tout_vec[i];
    tin[i] = tin_vec[i];
    tPredictedIn[i] = tPredictedIn_vec[i];
    tPredictedOut[i] = tPredictedOut_vec[i];
  }

  // initialize ON/OFF for predicted charging status, for each scenario,
  // charging station, and time period
  NumVarMatrix3D PredictedOnOffChargingStatus(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    PredictedOnOffChargingStatus[s] = NumVarMatrix2D(env, nbStations);
    for (int i = 0; i < nbStations; i++) {
      PredictedOnOffChargingStatus[s][i] =
          IloNumVarArray(env, nbTime, 0, 1, ILOINT);
    }
  }

  // initialize ON/OFF for real-time charging status, for each scenario,
  // charging station, and time period
  NumVarMatrix3D RealTimeOnOffChargingStatus(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    RealTimeOnOffChargingStatus[s] = NumVarMatrix2D(env, nbStations);
    for (int i = 0; i < nbStations; i++) {
      RealTimeOnOffChargingStatus[s][i] =
          IloNumVarArray(env, nbTime, 0, 1, ILOINT);
    }
  }

  // initialize Day-ahead charging power for $i$th EV, at $t$th interval, under
  // $s$th scenario, in kW
  NumVarMatrix3D RealTimeChargingPower(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    RealTimeChargingPower[s] = NumVarMatrix2D(env, nbStations);
    for (int i = 0; i < nbStations; i++) {
      RealTimeChargingPower[s][i] = IloNumVarArray(env, nbTime);
    }
  }

  // intialize predicted power output from utility grid at $t$th interval, under
  // $s$th scenario, in kW
  NumVarMatrix3D PredictedChargingPower(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    PredictedChargingPower[s] = NumVarMatrix2D(env, nbStations);
    for (int i = 0; i < nbStations; i++) {
      PredictedChargingPower[s][i] = IloNumVarArray(env, nbTime);
    }
  }

  // intialize Day-ahead power output from utility grid at $t$th interval, under
  // $s$th scenario, in kW
  NumVarMatrix2D RealTimeUtilityPowerOutput(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    RealTimeUtilityPowerOutput[s] = IloNumVarArray(env, nbTime, -IloInfinity, IloInfinity);
  }
  // initialize predicted power output from utility grid at $t$th interval,
  // under $s$th scenario, in kW
  NumVarMatrix2D PredictedUtilityPowerOutput(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    PredictedUtilityPowerOutput[s] = IloNumVarArray(env, nbTime, -IloInfinity, IloInfinity);
  }

  // intialize day-ahead buy/sell (1/0) status at $t$th interval, under $s$th
  // scenario
  NumVarMatrix2D RealTimeBuySellStatus(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    RealTimeBuySellStatus[s] = IloNumVarArray(env, nbTime, 0, 1, ILOINT);
  }

  // intialize day-ahead buy/sell (1/0) status at $t$th interval, under $s$th
  // scenario
  NumVarMatrix2D PredictedBuySellStatus(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    PredictedBuySellStatus[s] = IloNumVarArray(env, nbTime, 0, 1, ILOINT);
  }

  // initialize electricity purchase price from the grid at $t$th time interval
  // ($/kWh)
  IloNumArray ElectricityPurchasePrice(env, nbTime, 0, IloInfinity);

  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // *************** 2. Initialize variables for constraints ***************
  // ---------------------------------------------------------------------------

  // initialize Day-ahead solar output power in $t$th interval, under scenario
  // $s$ in kW
  NumVarMatrix2D RealTimeSolarOutput(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    RealTimeSolarOutput[s] = IloNumVarArray(env, nbTime, 0, IloInfinity);
  }

  // initialize state-of-charge (SOC) at $t$th interval, under scenario $s$, for
  // $i$th EV
  NumVarMatrix3D SOC(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    SOC[s] = NumVarMatrix2D(env, nbStations);
    for (int i = 0; i < nbStations; i++) {
      SOC[s][i] = IloNumVarArray(env, nbTime, 0, 1);
    }
  }

  std::vector<double> initialSOC_vec(nbStations);

  for (int i = 0; i < nbStations; i++) {
    initialSOC_vec[i] = std::round(norm_dist_out(rng));
  }

  for (int s = 0; s < nbScenarios; s++) {
    for (int i = 0; i < nbStations; i++) {
      model.add(SOC[s][i][tin[i]] == initialSOC_vec[i]);
    }
  }

  // variable to indicate whether the ith EV is parked during time period
  // [T_{in, i} , T_{out, i}]
  NumMatrix3D e(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    e[s] = NumMatrix2D(env, nbStations);
    for (int i = 0; i < nbStations; i++) {
      e[s][i] = IloNumArray(env, nbTime);
    }
  }

  // if T_{in, i} <= t0 && (T_{in, i} <= t <= T_{out, i}) then e = 1
  // if T_{in, i} > 0 && (Tpredicted_{in, i} <= t <= Tpredicted_{out, i}) then e
  // = 1 else e = 0
  for (int s = 0; s < nbScenarios; s++) {
    for (int i = 0; i < nbStations; i++) {
      for (int t = t0; t < t0 + H; t++) {
        if (tin[i] <= t0 && tin[i] <= t && t <= tout[i]) {
          e[s][i][t] = 1;
        } else if (tin[i] > 0 && tPredictedIn[i] <= t &&
                   t <= tPredictedOut[i]) {
          e[s][i][t] = 1;
        } else {
          e[s][i][t] = 0;
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // *************** 3. constraints ***************
  // ---------------------------------------------------------------------------

  // 1. Power balance
  // Total power output from solar and utility grid == Sum of EV charging load

  IloExprArray expr_powerbalance_predicted(env, nbScenarios);

  for (int s = 0; s < nbScenarios; s++) {
    expr_powerbalance_predicted[s] = RealTimeSolarOutput[s][t0] - RealTimeSolarOutput[s][t0];
    expr_powerbalance_predicted[s] += RealTimeSolarOutput[s][t0];
    expr_powerbalance_predicted[s] += RealTimeUtilityPowerOutput[s][t0];
    for (int i = 0; i < nbStations; i++) {
      expr_powerbalance_predicted[s] -= RealTimeChargingPower[s][i][t0];
    }
    model.add(expr_powerbalance_predicted[s] == 0);
  }

  ExprArray2D expr_powerbalance_realtime(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    expr_powerbalance_realtime[s] = IloExprArray(env, H);
  }

  for (int s = 0; s < nbScenarios; s++) {
    for (int t = t0 + 1; t < t0 + H; t++) {
      expr_powerbalance_realtime[s][t] += RealTimeSolarOutput[s][t];
      expr_powerbalance_realtime[s][t] += PredictedUtilityPowerOutput[s][t];
      for (int i = 0; i < nbStations; i++) {
        expr_powerbalance_realtime[s][t] -= PredictedChargingPower[s][i][t];
      }
      model.add(expr_powerbalance_realtime[s][t] == 0);
    }
  }

  // 2. Utility grid power output

  for (int s = 0; s < nbScenarios; s++) {
    model.add(RealTimeSolarOutput[s][t0] * (RealTimeBuySellStatus[s][t0] - 1) <=
              RealTimeUtilityPowerOutput[s][t0]);
    model.add(RealTimeUtilityPowerOutput[s][t0] <=
              Gamma * RealTimeBuySellStatus[s][t0]);
  }

  for (int s = 0; s < nbScenarios; s++) {
    for (int t = t0 + 1; t < t0 + H; t++) {
      model.add(RealTimeSolarOutput[s][t0] *
                    (PredictedBuySellStatus[s][t0] - 1) <=
                PredictedUtilityPowerOutput[s][t0]);
      model.add(PredictedUtilityPowerOutput[s][t0] <=
                Gamma * PredictedBuySellStatus[s][t0]);
    }
  }

  // 3. Charging power limit (part 1)
  // for each scenario, for each EV, SOC + total increasing SoC (which is
  // day-ahead charging power/Battery capacity) of $i$th EV across the whole
  // charging period == 1
  // for (int s = 0; s < nbScenarios; s++) {
  //   for (int i = 0; i < nbStations; i++) {
  //     IloExpr expr(env);
  //     expr += SOC[s][i][tHin[i]];
  //     for (int t = 0; t < nbHours; t++) {
  //       expr += RealTimeChargingPower[s][i][t] / BatteryCapacity[i];
  //     }
  //     model.add(expr == 1);
  //     expr.end();
  //   }
  // }

  // 4. Charging power limit (part 2)
  // Second and third constraint ensures that the charging power for every
  // scenario lies between the range $(0, P_{\text{level}}]$ when $\hat{u}_{s,
  // i}(t) == 1$. $\hat{P}_{s, i}(t) == 0 $ if $\hat{u}_{s, i}(t) == 0$.

  // for (int s = 0; s < nbScenarios; s++) {
  //   for (int i = 0; i < nbStations; i++) {
  //     for (int t = 0; t < nbHours; t++) {
  //       model.add(RealTimeChargingPower[s][i][t] <=
  //                 PredictedOnOffChargingStatus[s][i][t] * maxChargingPower);
  //       model.add(RealTimeChargingPower[s][i][t] >= 0);
  //     }
  //   }
  // }

  // 5. Charging power limit (part 3)
  // \varepsilon \hat{u}_{s, i}(t) & \leq \hat{P}_{s, i}(t) & \leq P_{\text
  // {level }} \hat{u}_{s, i}(t) & \forall s, \forall i, \forall t
  for (int s = 0; s < nbScenarios; s++) {
    for (int i = 0; i < nbStations; i++) {
      model.add(RealTimeChargingPower[s][i][t0] <=
                RealTimeOnOffChargingStatus[s][i][t0] * maxChargingPower);
      model.add(RealTimeChargingPower[s][i][t0] >=
                Epsilon * RealTimeOnOffChargingStatus[s][i][t0]);
    }
  }

  for (int s = 0; s < nbScenarios; s++) {
    for (int i = 0; i < nbStations; i++) {
      for (int t = t0 + 1; t < t0 + H; t++) {
        model.add(PredictedChargingPower[s][i][t] <=
                  PredictedOnOffChargingStatus[s][i][t] * maxChargingPower);
        model.add(RealTimeChargingPower[s][i][t] >=
                  Epsilon * PredictedOnOffChargingStatus[s][i][t]);
      }
    }
  }

  // Limitation of SOC
  for (int s = 0; s < nbScenarios; s++) {
    for (int i = 0; i < nbStations; i++) {
      for (int t = 0; t < nbTime; t++) {
        model.add(SOC[s][i][t] <= SOC_max);
        model.add(SOC[s][i][t] >= 0);
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
  IloExpr expr_fParking(env);
  for (int s = 0; s < nbScenarios; s++) {
    for (int i = 0; i < nbStations; i++) {
      expr_fParking += (e[s][i][t0]) * timeInterval -
                       (rebaterate * timeInterval * prob[s] *
                        PredictedOnOffChargingStatus[s][i][t0]);
    }
  }
  expr_fParking = rparking * expr_fParking;

  // define expression for \hat{R}_{\text {parking }}=r_{\text {parking }}
  // \sum_{i=1}^N \hat{t}_{\text {parking }, i}
  IloExpr expr_fPredictedParking(env);
  for (int s = 0; s < nbScenarios; s++) {
    for (int t = t0 + 1; t < t0 + H; t++) {
      for (int i = 0; i < nbStations; i++) {
        expr_fPredictedParking += (e[s][i][t]) * timeInterval -
                                  (rebaterate * timeInterval * prob[s] *
                                   PredictedOnOffChargingStatus[s][i][t]);
      }
    }
  }
  expr_fPredictedParking = rparking * expr_fPredictedParking;

  // define expression for expected day-ahead charging revenue
  // \hat{R}_{\text {charging }}=\sum_{s=1}^S
  // \operatorname{prob}_s \sum_{t=1}^T[ & r_{\text {charging }} \sum_{i=1}^N
  // \hat{P}_{s, i}(t) \Delta t -\hat{P}_{\text {grid }, s}(t) \hat{y}_s(t)
  // c(t) \Delta t - \hat{P}_{\text {grid }, s} (1-\hat{y}_s(t)) h(t) \Delta t

  IloExpr expr_fCharging(env);
  for (int s = 0; s < nbScenarios; s++) {
    IloExpr expr_fCharging_i(env);
    for (int i = 0; i < nbStations; i++) {
      expr_fCharging_i +=
          (rcharging * RealTimeChargingPower[s][i][t0] * timeInterval -
           RealTimeUtilityPowerOutput[s][t0] * RealTimeBuySellStatus[s][t0] *
               TOUPurchasePrice[t0] * timeInterval -
           RealTimeUtilityPowerOutput[s][t0] *
               (1 - RealTimeBuySellStatus[s][t0]) * TOUSellPrice[t0] *
               timeInterval);
    }
    expr_fCharging += prob[s] * expr_fCharging_i;
  }

  IloExpr expr_fPredictedCharging(env);
  for (int s = 0; s < nbScenarios; s++) {
    for (int t = t0 + 1; t < t0 + H; t++) {
      IloExpr expr_fPredictedCharging_i(env);
      for (int i = 0; i < nbStations; i++) {
        expr_fPredictedCharging_i +=
            (rcharging * RealTimeChargingPower[s][i][t] * timeInterval -
             RealTimeUtilityPowerOutput[s][t] * RealTimeBuySellStatus[s][t] *
                 TOUPurchasePrice[t] * timeInterval -
             RealTimeUtilityPowerOutput[s][t] *
                 (1 - RealTimeBuySellStatus[s][t]) * TOUSellPrice[t] *
                 timeInterval);
      }
      expr_fPredictedCharging += prob[s] * expr_fPredictedCharging_i;
    }
  }

  // define expression for \hat{R} = \hat{R}_{\text {charging }} +
  // \hat{R}_{\text {parking }} - C_{\text {operation }}
  IloExpr expr_f(env);
  expr_f = expr_fParking + expr_fCharging + expr_fPredictedParking +
           expr_fPredictedCharging;

  model.add(IloMaximize(env, expr_f));
  expr_f.end();
  expr_fCharging.end();
  expr_fParking.end();
  expr_fPredictedCharging.end();
  expr_fPredictedParking.end();
  // expr_powerbalance_predicted.end();
  expr_powerbalance_realtime.end();

  return 0;
}

// void matread(const char *file, const char *name, std::vector<double> &v) {
//   MATFile *pmat;
//   mxArray *pa;
//   pmat = matOpen(file, "r");
//   if (pmat == NULL) {
//     printf("Error opening file %s", file);
//     return;
//   }

//   pa = matGetVariable(pmat, name);
//   if (pa != NULL && mxIsDouble(pa) && !mxIsEmpty(pa)) {
//     // copy data
//     mwSize num = mxGetNumberOfElements(pa);
//     double *pr = mxGetPr(pa);
//     if (pr != NULL) {
//       v.reserve(num); // is faster than resize :-)
//       v.assign(pr, pr + num);
//     }
//   }

//     // cleanup
//   mxDestroyArray(pa);
//   matClose(pmat);
// }