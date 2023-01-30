# list files in the current directory and go through each one
for file in ~/Documents/Optimization/DayAheadForecast/Data/scenarios/*; do
    # print the file name
    echo "The filename is:"
    echo $file

    echo "Running the program..."
    ./build/DayAheadForecast $file

    # wait for 5 seconds
    sleep 5
done