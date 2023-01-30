# run generate_solar_scenarios.py 30 times with argument 100, parse i as the counter argument

for i in {1..30}
do
    python generate_solar_scenarios.py 100 $i
done