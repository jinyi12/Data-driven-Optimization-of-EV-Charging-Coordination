// initial cpp version of day-ahead forecast model.
// This script is only for reference, and it is not used in the final version of the data-driven day-ahead forecast model.
// Please refer to the python version of the day-ahead forecast model for the final version.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ilconcert/ilocsvreader.h>
#include <ilcplex/ilocplex.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ilconcert/iloenv.h"
#include "ilconcert/iloexpression.h"
#include "ilconcert/ilosys.h"
#include "ilcplex/ilocplexi.h"

// #include <highfive/H5Easy.hpp>

ILOSTLBEGIN

typedef IloArray<IloNumVarArray> NumVarMatrix2D;
typedef IloArray<NumVarMatrix2D> NumVarMatrix3D;
typedef IloArray<IloNumArray> NumMatrix2D;
typedef IloArray<NumMatrix2D> NumMatrix3D;
typedef IloArray<IloExprArray> ExprArray2D;
typedef IloArray<ExprArray2D> ExprArray3D;

// function to read csv file into a std::vector
std::vector<std::vector<double>> read_csv(const std::string &filename,
                                          bool header);

int main(int argc, char *argv[]) {
  printf("Hello World!");

  //  if not enough arguments, use the default input file
  if (argc < 2) {
    std::cout << "Not enough arguments, use the default input file"
              << std::endl;
    argv[1] = "/Users/jinyiyong/Documents/Optimization/DayAheadForecast/Data/"
              "scenarios/scenarios_1.csv";
  }

  // assign input file name
  std::string input_file = argv[1];

  cout << "Input file: " << input_file << endl;

  int nbScenarios = 100;

  IloEnv env;
  IloModel model(env);
  IloCplex cplex(env);

  // In the first stage simulation, rparking is equal to $1.5/per hour, and the
  // fixed daily operational cost is equal to $4000 per 24 h. The number of
  // charging stations in this EV parking deck is assumed to be 300.

  IloNum rparking = 1.5;
  IloNum fixedcost = 4000;
  IloInt nbStations = 300;

  IloInt nbTime = 48;
  IloNum timeInterval = 0.5;

  IloNum Gamma = 100000;
  IloNum Epsilon = 0.00001;

  // maximum charging power of each EV, in kW, using level 2 charging
  IloNum maxChargingPower = 19;

  IloNum rebaterate = 0.6;
  // IloNumVar rcharging(env, 0, IloInfinity, ILOFLOAT);
  IloNum rcharging = 1.5;

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
  IloNumArray TOUPurchasePrice(env, nbTime, 0, IloInfinity, ILOFLOAT);
  IloNumArray TOUSellPrice(env, nbTime, 0, IloInfinity, ILOFLOAT);

  // 08:00 to 11:00 mid-peak, 11:00 to 12:00 peak, 12:00 to 14:00 mid-peak,
  // 14:00 to 17:00 peak, 17:00 to 22:00 mid peak, 22:00 to 08:00 off-peak
  // TOU purchase price is 0.584/kWh on peak, 0.357 on mid-peak, and 0.281 on
  // off peak
  // TOU sell price is 10% cheaper than TOU purchase price
  // According to the above comments, create TOU purchase price and TOU sell
  // price arrays
  // 48 time intervals, each time interval is 30 minutes

  float off_peak = 0.281;
  float mid_peak = 0.357;
  float peak = 0.584;

  for (int t = 0; t < nbTime; t++) {
    if (t >= 0 && t < 16) {
      TOUPurchasePrice[t] = off_peak;
      TOUSellPrice[t] = 0.9 * off_peak;
    }
    if (t >= 16 && t < 22) {
      TOUPurchasePrice[t] = mid_peak;
      TOUSellPrice[t] = 0.9 * mid_peak;
    }
    if (t >= 22 && t < 24) {
      TOUPurchasePrice[t] = peak;
      TOUSellPrice[t] = 0.9 * peak;
    }
    if (t >= 24 && t < 28) {
      TOUPurchasePrice[t] = mid_peak;
      TOUSellPrice[t] = 0.9 * mid_peak;
    }
    if (t >= 28 && t < 34) {
      TOUPurchasePrice[t] = peak;
      TOUSellPrice[t] = 0.9 * peak;
    }
    if (t >= 34 && t < 40) {
      TOUPurchasePrice[t] = mid_peak;
      TOUSellPrice[t] = 0.9 * mid_peak;
    }
    if (t >= 40 && t < 48) {
      TOUPurchasePrice[t] = off_peak;
      TOUSellPrice[t] = 0.9 * off_peak;
    }
  }

  // ---------------------------------------------------------------------------
  // *************** 1. Initialize variables for objective function
  // ***************
  // ---------------------------------------------------------------------------

  // initialize predicted time out and time in variables, for each charging
  // station
  IloIntArray tHout(env, nbStations, 0, nbTime, ILOINT);
  IloIntArray tHin(env, nbStations, 0, nbTime, ILOINT);

  std::vector<double> tHout_vec(nbStations);
  std::vector<double> tHin_vec(nbStations);
  std::vector<std::vector<double>> SOC_vec_2D(nbStations);
  std::vector<double> SOC_vec(nbStations);

  // read arrival and departure .csv files located in the data folder and
  // initialize tHout and tHin, tPredicterIn and tPredictedOut
  std::filesystem::path current_path = std::filesystem::current_path();
  std::cout << "Current path is : " << current_path << std::endl;

  // if the current path is not the root path, then go to parent path
  if (current_path.filename() != "DayAheadForecast") {
    current_path = current_path.parent_path();
  }

  std::cout << "Current path now is : " << current_path << std::endl;

  const char *arrival_file = "Data/arrival_departure_times.csv";
  std::filesystem::path arrival_path = current_path / arrival_file;
  const char *arrival_path_char = arrival_path.c_str();

  // read arrival and departure times from .csv file
  std::vector<std::vector<double>> arrival_departure_vec_2D =
      read_csv(arrival_path_char, true);

  // first column is arrival time, second column is departure time
  for (int i = 0; i < nbStations; i++) {
    tHout_vec[i] = arrival_departure_vec_2D[i][1];
    tHout[i] = round(tHout_vec[i] / timeInterval);

    tHin_vec[i] = arrival_departure_vec_2D[i][0];
    tHin[i] = round(tHin_vec[i] / timeInterval);
  }

  // read SOC distribution from .csv
  const char *SOC_file = "Data/SOC.csv";
  std::filesystem::path SOC_path = current_path / SOC_file;
  const char *SOC_path_char = SOC_path.c_str();

  SOC_vec_2D = read_csv(SOC_path_char, false);
  // if each vector entry is of size 1, then it is a 1D vector, convert it to a
  // 1D vec
  if (SOC_vec_2D[0].size() == 1) {
    for (int i = 0; i < SOC_vec_2D.size(); i++) {
      SOC_vec[i] = SOC_vec_2D[i][0];
    }
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
      DayAheadChargingPower[s][i] =
          IloNumVarArray(env, nbTime, 0, maxChargingPower, ILOFLOAT);
    }
  }

  // intialize Day-ahead power output from utility grid at $t$th interval, under
  // $s$th scenario, in kW
  NumVarMatrix2D DayAheadUtilityPowerOutput(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    DayAheadUtilityPowerOutput[s] =
        IloNumVarArray(env, nbTime, -IloInfinity, IloInfinity, ILOFLOAT);
  }

  // intialize day-ahead buy/sell (1/0) status at $t$th interval, under $s$th
  // scenario
  NumVarMatrix2D DayAheadBuySellStatus(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    DayAheadBuySellStatus[s] = IloNumVarArray(env, nbTime, 0, 1, ILOINT);
  }

  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // *************** 2. Initialize variables for constraints ***************
  // ---------------------------------------------------------------------------

  // initialize Day-ahead solar output power in $t$th interval, under scenario
  // $s$ in kW
  // read csv file for solar scenarios

  // const char *solar_file = "Data/scenarios.csv";
  // std::filesystem::path solar_path = current_path / solar_file;
  // const char *solar_path_char = solar_path.c_str();

  // convert string
  // input_file="/Users/jinyiyong/Documents/Optimization/DayAheadForecast/Data/scenarios/scenarios_12.csv"
  // to char*

  cout << "extracting number from file name" << endl;
  const char *solar_path_char = input_file.c_str();

  // print solar_path_char
  cout << "solar_path_char is " << solar_path_char << endl;

  // extract the digit from file name
  std::string token1 = input_file.substr(input_file.find_last_of("_") + 1);
  std::string token2 = token1.substr(0, token1.find_last_of("."));
  int fileNumber = std::stoi(token2);
  cout << "file number is " << fileNumber << endl;

  std::string output_file = (current_path / "Data/output/output_");
  output_file = output_file + std::to_string(fileNumber) + ".csv";
  cout << "output file is " << output_file << endl;

  std::vector<std::vector<double>> solar_scenarios_2D =
      read_csv(solar_path_char, false);
  NumMatrix2D DayAheadSolarOutput(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    DayAheadSolarOutput[s] = IloNumArray(env, nbTime, 0, IloInfinity);
    for (int t = 0; t < nbTime; t++) {
      DayAheadSolarOutput[s][t] = solar_scenarios_2D[s][t];
    }
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

  ExprArray2D PowerBalance(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    PowerBalance[s] = IloExprArray(env, nbTime);
    for (int t = 0; t < nbTime; t++) {
      PowerBalance[s][t] = IloExpr(env);
      PowerBalance[s][t] += DayAheadSolarOutput[s][t];
      PowerBalance[s][t] += DayAheadUtilityPowerOutput[s][t];
      for (int i = 0; i < nbStations; i++) {
        PowerBalance[s][t] -= DayAheadChargingPower[s][i][t];
      }
      model.add(PowerBalance[s][t] == 0);
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

  //  ExprArray2D ChargingPowerLimit1(env, nbScenarios);
  //  for (int s = 0; s < nbScenarios; s++) {
  //    ChargingPowerLimit1[s] = IloExprArray(env, nbStations);
  //    for (int i = 0; i < nbStations; i++) {
  //      ChargingPowerLimit1[s][i] = IloExpr(env);
  //      int arrivaltime = tHin[i];
  //      ChargingPowerLimit1[s][i] += SOC[s][i][arrivaltime];
  //      for (int t = arrivaltime; t <= tHout[i]; t++) {
  //        ChargingPowerLimit1[s][i] +=
  //            DayAheadChargingPower[s][i][t] * timeInterval /
  //            BatteryCapacity[i];
  //      }
  //      model.add(ChargingPowerLimit1[s][i] == 1);
  //    }
  //  }

  // initial SOC
  for (int s = 0; s < nbScenarios; s++) {
    for (int i = 0; i < nbStations; i++) {

      for (int t = 0; t < nbTime; t++) {
        if (t <= tHin[i]) {
          model.add(SOC[s][i][t] == SOC_vec[i]);
        }
        //        else if (t > tHout[i]) {
        //          model.add(SOC[s][i][t] == SOC[s][i][tHout[i]]);
        //        }
        if (t > tHin[i] && t <= tHout[i]) {
          model.add(SOC[s][i][t] ==
                    SOC[s][i][t - 1] + (DayAheadChargingPower[s][i][t] *
                                        timeInterval / BatteryCapacity[i]));
        }

        if (t > tHout[i]) {
          model.add(SOC[s][i][t] == SOC[s][i][tHout[i]]);
        }
      }

      IloExpr expr(env);
      expr += SOC[s][i][tHin[i]];

      for (int t = 0; t < nbTime; t++) {
        expr +=
            DayAheadChargingPower[s][i][t] * timeInterval / BatteryCapacity[i];
      }

      model.add(expr == 1);
      expr.end();
    }
  }

  // 4. Charging power limit (part 2)
  // Second and third constraint ensures that the charging power for every
  // scenario lies between the range $(0, P_{\text{level}}]$ when
  // $\hat{u}_{s, i}(t) == 1$. $\hat{P}_{s, i}(t) == 0 $ if $\hat{u}_{s,
  // i}(t) == 0$.

  for (int s = 0; s < nbScenarios; s++) {
    for (int i = 0; i < nbStations; i++) {
      for (int t = 0; t < nbTime; t++) {
        if (t < tHin[i] || t > tHout[i]) {
          model.add(DayAheadOnOffChargingStatus[s][i][t] == 0);
        }

        model.add(DayAheadChargingPower[s][i][t] <=
                  DayAheadOnOffChargingStatus[s][i][t] * maxChargingPower);
        model.add(DayAheadChargingPower[s][i][t] >=
                  Epsilon * DayAheadOnOffChargingStatus[s][i][t]);
      }
    }
  }

  // Linearize DayAheadChargingPower * DayAheadOnOffChargingStatus
  NumVarMatrix3D DayAheadChargingPowerLinearized(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    DayAheadChargingPowerLinearized[s] = NumVarMatrix2D(env, nbStations);
    for (int i = 0; i < nbStations; i++) {
      DayAheadChargingPowerLinearized[s][i] =
          IloNumVarArray(env, nbTime, 0, IloInfinity, ILOFLOAT);
      for (int t = 0; t < nbTime; t++) {
        model.add(DayAheadChargingPowerLinearized[s][i][t] <=
                  maxChargingPower * DayAheadOnOffChargingStatus[s][i][t]);
        model.add(DayAheadChargingPowerLinearized[s][i][t] <=
                  DayAheadChargingPower[s][i][t]);
        model.add(DayAheadChargingPowerLinearized[s][i][t] >= 0);
        model.add(DayAheadChargingPowerLinearized[s][i][t] >=
                  DayAheadChargingPower[s][i][t] -
                      (1 - DayAheadOnOffChargingStatus[s][i][t]) *
                          maxChargingPower);
      }
    }
  }

  // Linearize DayAheadUtilityPowerOutput * DayAheadBuySellStatus
  NumVarMatrix2D DayAheadUtilityPowerOutputLinearizedPurchase(env, nbScenarios);
  for (int s = 0; s < nbScenarios; s++) {
    DayAheadUtilityPowerOutputLinearizedPurchase[s] =
        IloNumVarArray(env, nbTime, -IloInfinity, IloInfinity, ILOFLOAT);
    for (int t = 0; t < nbTime; t++) {

      model.add(DayAheadUtilityPowerOutputLinearizedPurchase[s][t] <=
                Gamma * DayAheadBuySellStatus[s][t]);
      model.add(DayAheadUtilityPowerOutputLinearizedPurchase[s][t] >=
                -Gamma * DayAheadBuySellStatus[s][t]);

      model.add(DayAheadUtilityPowerOutputLinearizedPurchase[s][t] <= Gamma);
      model.add(DayAheadUtilityPowerOutputLinearizedPurchase[s][t] >= -Gamma);

      model.add(DayAheadUtilityPowerOutputLinearizedPurchase[s][t] >=
                DayAheadUtilityPowerOutput[s][t] -
                    (1 - DayAheadBuySellStatus[s][t]) * Gamma);
      model.add(DayAheadUtilityPowerOutputLinearizedPurchase[s][t] <=
                DayAheadUtilityPowerOutput[s][t] -
                    (1 - DayAheadBuySellStatus[s][t]) * -Gamma);

      model.add(DayAheadUtilityPowerOutputLinearizedPurchase[s][t] <=
                DayAheadUtilityPowerOutput[s][t] +
                    (1 - DayAheadBuySellStatus[s][t]) * Gamma);
    }
  }

  // 5. Charging power limit (part 3)
  // \varepsilon \hat{u}_{s, i}(t) & \leq \hat{P}_{s, i}(t) & \leq P_{\text
  // {level }} \hat{u}_{s, i}(t) & \forall s, \forall i, \forall t
  //  for (int s = 0; s < nbScenarios; s++) {
  //    for (int i = 0; i < nbStations; i++) {
  //      for (int t = 0; t < nbTime; t++) {
  //        model.add(DayAheadChargingPower[s][i][t] <=
  //                  DayAheadOnOffChargingStatus[s][i][t] * maxChargingPower);
  //        model.add(DayAheadChargingPower[s][i][t] >=
  //                  Epsilon * DayAheadOnOffChargingStatus[s][i][t]);
  //      }
  //    }
  //  }

  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // *************** 4. objective function ***************
  // ---------------------------------------------------------------------------

  // The sum of the differences between the time of EV exit
  // ($\hat{T}_{\text{out}, i}$) and the time of EV entry
  // ($\hat{T}_{\text{in}, i}$) for all $i$ from 1 to $N$, and then subtract
  // parking fee rebate rate * time interval * prob[s] *
  // DayAheadOnOffChargingStatus .
  IloExpr expr_tHatParking(env);
  for (int s = 0; s < nbScenarios; s++) {
    for (int i = 0; i < nbStations; i++) {
      expr_tHatParking += (tHout[i] - tHin[i]);
      for (int t = 0; t < nbTime; t++) {
        expr_tHatParking -= (rebaterate * timeInterval * prob[s] *
                             DayAheadOnOffChargingStatus[s][i][t]);
      }
    }
  }

  // define expression for \hat{R}_{\text {parking }}=r_{\text {parking }}
  // \sum_{i=1}^N \hat{t}_{\text {parking }, i}
  IloExpr expr_RHatParking(env);
  expr_RHatParking = rparking * expr_tHatParking;

  IloExpr expr_RHatCharging(env);
  for (int s = 0; s < nbScenarios; s++) {
    for (int t = 0; t < nbTime; t++) {
      for (int i = 0; i < nbStations; i++) {
        expr_RHatCharging +=
            prob[s] * (rcharging * DayAheadChargingPowerLinearized[s][i][t] *
                       timeInterval);
      }
      //      expr_RHatCharging -= prob[s] * (DayAheadUtilityPowerOutput[s][t] *
      //                                      DayAheadBuySellStatus[s][t] *
      //                                      TOUPurchasePrice[t] *
      //                                      timeInterval);
      //      expr_RHatCharging -= prob[s] * (DayAheadUtilityPowerOutput[s][t] *
      //                                      (1 - DayAheadBuySellStatus[s][t])
      //                                      * TOUSellPrice[t] * timeInterval);

      expr_RHatCharging -=
          prob[s] * (DayAheadUtilityPowerOutputLinearizedPurchase[s][t] *
                     TOUPurchasePrice[t] * timeInterval);
      expr_RHatCharging -=
          prob[s] * ((DayAheadUtilityPowerOutput[s][t] -
                      DayAheadUtilityPowerOutputLinearizedPurchase[s][t]) *
                     TOUSellPrice[t] * timeInterval);
    }
  }
  // define expression for \hat{R} = \hat{R}_{\text {charging }} +
  // \hat{R}_{\text {parking }} - C_{\text {operation }}
  IloExpr expr_RHat(env);
  expr_RHat = expr_RHatParking + expr_RHatCharging - fixedcost;

  model.add(IloMaximize(env, expr_RHat));
  expr_RHat.end();
  expr_RHatParking.end();
  expr_RHatCharging.end();
  expr_tHatParking.end();
  PowerBalance.end();

  try {
    cplex.extract(model);
  } catch (IloException &e) {
    std::cerr << "Concert exception caught: " << e << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception caught" << std::endl;
  }

  // CPLEX parameter setting :
  // cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, 0.01);  //
  // setting the optimal gap to be 0.01%
  //  cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, 0.03);
  // cplex.setParam(IloCplex::Param::MIP::Strategy::MIQCPStrat, 2);   //
  // various setting, you can check for them by google IBM cplex user manual
  // cplex.setParam(IloCplex::Param::Parallel, -1);
  // cplex.setParam(IloCplex::Param::RootAlgorithm, 1);
  // cplex.setParam(IloCplex::Param::MIP::Display, 3);
  // cplex.setParam(IloCplex::Param::TimeLimit, 3600);  //time limit to
  // 3600s
  cplex.setParam(IloCplex::Param::Threads,
                 1); // using one thread or core of CPU
  cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, 0.03);
  cplex.setParam(IloCplex::Param::RootAlgorithm, 1);

  try {
    cplex.solve();
  } catch (IloException &e) {
    cerr << "Concert exception caught: " << e << endl;
  } catch (...) {
    cerr << "Unknown exception caught" << endl;
  }

  cout << setprecision(12);
  cout << "Solution status: " << cplex.getStatus() << endl;
  cout << "Cplex Obj = " << cplex.getObjValue() << endl;

  // output the optimal solution
  ofstream myfile;
  myfile.open(output_file);
  myfile << "Scenario,Time,tHin_i,tHout_i,i,DayAheadBuySellStatus,"
            "DayAheadOnOffChargingStatus,"
            "DayAheadChargingPower,DayAheadUtilityPowerOutput,SOC"
         << endl;

  for (int s = 0; s < nbScenarios; s++) {
    for (int t = 0; t < nbTime; t++) {
      for (int i = 0; i < nbStations; i++) {
        myfile << s << "," << t << "," << tHin[i] << "," << tHout[i] << "," << i
               << "," << cplex.getValue(DayAheadBuySellStatus[s][t]) << ","
               << cplex.getValue(DayAheadOnOffChargingStatus[s][i][t]) << ","
               << cplex.getValue(DayAheadChargingPower[s][i][t]) << ","
               << cplex.getValue(DayAheadUtilityPowerOutput[s][t]) << ","
               << cplex.getValue(SOC[s][i][t]) << endl;
      }
    }
  }

  // if (cplex.getStatus() == IloAlgorithm::Optimal) {
  //   IloConstraint basis = cplex.getBasisStatuses();
  //   std::vector<int> active_constraints;
  //   for (int i = 0; i < cplex.getNrows(); i++) {
  //     if (basis.getBasisStatus(i) == IloBasisStatus::Basic) {
  //       active_constraints.push_back(i);
  //     }
  //   }
  //   // Store active constraints in active_constraints
  //   ...
  // }

  // try { // basis may not exist
  // IloCplex::BasisStatusArray cstat(env);
  // cplex.getBasisStatuses(cstat, var);
  // env.out() << "Basis statuses = " << cstat << endl;
  // } catch (...) {
  // }

  // output the active sets
  ofstream myfile2;
  std::string output_file2 =
      output_file + std::to_string(fileNumber) + "_activeSet.csv";
  myfile2.open(output_file2);
  myfile2 << "Scenario,Time,i,activeSet_DayAheadBuySellStatus,activeSet_"
             "DayAheadOnOffChargingStatus,activeSet_"
             "DayAheadChargingPower"
          << endl;


  
  

  for (int s = 0; s < nbScenarios; s++) {
    for (int t = 0; t < nbTime; t++) {
      for (int i = 0; i < nbStations; i++) {
        // basis might not exist

        try {

          
          IloCplex::BasisStatus basisStatus;
          basisStatus = cplex.getBasisStatus(DayAheadChargingPower[s][i][t]);

          

          env.out() << "Basis status = " << basisStatus << endl;

          // myfile2 << s << "," << t << "," << i << ","
          //         << cplex.getBasisStatus(DayAheadBuySellStatus[s][t]) << ","
          //         <<
          //         cplex.getBasisStatus(DayAheadOnOffChargingStatus[s][i][t])
          //         << ","
          //         << cplex.getBasisStatus(DayAheadChargingPower[s][i][t])
          //         << endl;
        } catch (IloException &e) {
          cerr << "Concert exception caught: " << e << endl;
        } catch (...) {
          cerr << "Unknown exception caught" << endl;
        }
      }
    }
  }

  return 0;
}

std::vector<std::vector<double>> read_csv(const std::string &filename,
                                          bool header = false) {
  std::vector<std::vector<double>> data;
  std::ifstream file(filename);
  std::string line;

  if (header) {

    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    // skip the first line

    while (std::getline(file, line)) {
      std::stringstream lineStream(line);
      std::string cell;
      std::vector<double> row;

      while (std::getline(lineStream, cell, ',')) {
        row.push_back(std::stod(cell));
      }
      data.push_back(row);
    }

  } else {
    while (std::getline(file, line)) {
      std::stringstream lineStream(line);
      std::string cell;
      std::vector<double> row;

      while (std::getline(lineStream, cell, ',')) {
        row.push_back(std::stod(cell));
      }
      data.push_back(row);
    }
  }
  return data;
}