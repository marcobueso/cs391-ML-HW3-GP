

import csv
from matplotlib import pyplot as plt

with open(r"C:\Users\boysb\OneDrive\Documentos\UTexas\Fall 2021\Machine Learning\cs391-ML-HW3-GP\data_GP\AG\block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM\20161213203046-59968-right-speed_0.500.csv") as csvfile:
    sensor_data = []
    sensor_4_data = []
    time = []
    reader = csv.reader(csvfile)
    header = next(reader)
    for row in reader:
        time.append(row[1])
        number_sensors = 10
        d_sensors = {}
        for i in range(number_sensors):
            d_sensors[i] = []
        for i in range(11, 11+number_sensors*4 , 4):
            if float(row[i+3]) > 0: # c value
                print("Adding to sensor: ", (i-11) / 4 )
                d_sensors[int( (i-11) / 4)] = [ float(row[i]), float(row[i+1]), float(row[i+2]) ]
                #sensor_4_data.append([float(row[28]), float(row[29]), float(row[30])])

print(d_sensors[3])
#plt.plot(time, d_sensors[3])
#plt.show()