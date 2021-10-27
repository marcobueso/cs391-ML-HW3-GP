

import csv
from matplotlib import pyplot as plt
import numpy as np

### load_data ###
# Loads the data from csv file (path), and stores into dictionary d_sensors
# Loadtime only true once, as all sensors share time
def load_data(path, d_sensors, num_sensors = 50, loadTime = False):
    with open(path) as csvfile:
        sensor_data = []    
        number_sensors = num_sensors
        for i in range(number_sensors):
            #d_sensors[i] = np.empty(shape=(1030,3))
            d_sensors[i] = []
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            if loadTime:
                time.append(row[1])

            for i in range(11, 11+number_sensors*4 , 4):
                if float(row[i+3]) > 0: # c value
                    d_sensors[int( (i-11) / 4)].append( [ float(row[i]), float(row[i+1]), float(row[i+2]) ] )
                else: #if int( (i-11) / 4) == 0: # handle initial case
                    d_sensors[int( (i-11) / 4)].append( [0,0,0] )
                # else: # append previous value
                #     d_sensors[int( (i-11) / 4)].append( d_sensors[int( (i-11) / 4) - 1]  )

# code from StackOverflow user: Martin Geisler, https://stackoverflow.com/questions/903853/how-do-you-extract-a-column-from-a-multi-dimensional-array
def column(matrix, i):
    return [row[i] for row in matrix]

# Initialize time, and dictionaries for 5 rounds of movement by one subject 
time = []
d1_sensors = {}
d2_sensors = {}
d3_sensors = {}
d4_sensors = {}
d5_sensors = {}

load_data(r"C:\Users\boysb\OneDrive\Documentos\UTexas\Fall 2021\Machine Learning\cs391-ML-HW3-GP\data_GP\AG\block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM\20161213203046-59968-right-speed_0.500.csv", d1_sensors, num_sensors = 10, loadTime = True)
load_data(r"C:\Users\boysb\OneDrive\Documentos\UTexas\Fall 2021\Machine Learning\cs391-ML-HW3-GP\data_GP\AG\block2-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM\20161213204004-59968-right-speed_0.500.csv", d2_sensors, num_sensors = 10)
load_data(r"C:\Users\boysb\OneDrive\Documentos\UTexas\Fall 2021\Machine Learning\cs391-ML-HW3-GP\data_GP\AG\block3-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM\20161213204208-59968-right-speed_0.500.csv", d3_sensors, num_sensors = 10)
load_data(r"C:\Users\boysb\OneDrive\Documentos\UTexas\Fall 2021\Machine Learning\cs391-ML-HW3-GP\data_GP\AG\block4-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM\20161213204925-59968-right-speed_0.500.csv", d4_sensors, num_sensors = 10)
load_data(r"C:\Users\boysb\OneDrive\Documentos\UTexas\Fall 2021\Machine Learning\cs391-ML-HW3-GP\data_GP\AG\block5-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM\20161213210121-59968-right-speed_0.500.csv", d5_sensors, num_sensors = 10)


# plt.plot(time, column(d1_sensors[0],1),time,column(d2_sensors[0],1),time,column(d3_sensors[0],1),time,column(d4_sensors[0],1),time,column(d5_sensors[0],1))
# plt.show()
# plt.plot(time, column(d1_sensors[0],1))
# plt.show()
# print(column(d1_sensors[0],1))
# Display some eigenvectors
# fig = plt.figure(figsize=(10, 10))
# columns = 5
# rows = 5
# for i in range(1, columns*rows +1):
#     fig.add_subplot(rows, columns, i)
#     plt.plot(time, column(d1_sensors[i],1))
# plt.show()
#print(column(d1_sensors[1],1))
num_side = 3
fig, axs = plt.subplots(num_side, num_side)
for i in range(num_side):
    for j in range(num_side):
        print(i,j)
        axs[i,j].plot(time, column(d1_sensors[i+j],1))
        axs[i,j].axis('off')
    

plt.show()