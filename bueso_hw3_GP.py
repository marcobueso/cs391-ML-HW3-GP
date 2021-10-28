
import csv
from matplotlib import pyplot as plt
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

## PLOTTING FLAGS ##
plot1 = False

### load_data ###
# Loads the data from csv file (path), and stores into dictionary d_sensors
# Loadtime only true once, as all sensors share time
def load_data(path, d_sensors, num_sensors = 50, loadTime = False):
    with open(path) as csvfile:
        sensor_data = []    
        number_sensors = num_sensors
        for i in range(number_sensors):
            d_sensors[i] = []
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            if loadTime:
                time.append(row[1])

            for i in range(11, 11+number_sensors*4 , 4):
                if float(row[i+3]) > 0: # c value
                    d_sensors[int( (i-11) / 4)].append( [ float(row[i]), float(row[i+1]), float(row[i+2]) ] )
                elif int( (i-11) / 4) == 0: # handle initial case
                    d_sensors[int( (i-11) / 4)].append( [0,0,0] )
                else: # append previous value
                    d_sensors[int( (i-11) / 4)].append( d_sensors[int( (i-11) / 4) - 1]  )

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



### PLOT first num_side^2 sensors' data - the 'y' component
# num_side = 3
# fig, axs = plt.subplots(num_side, num_side)
# for i in range(num_side):
#     for j in range(num_side):
#         print(i,j)
#         axs[i,j].plot(time, column(d1_sensors[i+j],1))
#         axs[i,j].axis('off')
# plt.show()


### PLOT sensor 5's 'y' component data, for 4 trials
col = 1
if plot1:
    plt.plot(time,column(d2_sensors[5],col),time,column(d3_sensors[5],col),time,column(d4_sensors[5],col),time,column(d5_sensors[5],col))
    plt.show()


## GAUSSIAN PROCESS
my_kernel = C(1.0) * RBF(1030) + WhiteKernel()
GP = GaussianProcessRegressor(kernel = my_kernel)

X = np.atleast_2d(range(len(time))).T

GP.fit(X,column(d2_sensors[5],col))

x = np.atleast_2d(np.linspace(1, 1030, 1030)).T

y_pred, sigma = GP.predict(x, return_std=True)
plt.figure()
plt.plot(X, column(d2_sensors[5],col), "r-", markersize=2, label="Observations")
plt.plot(x, y_pred, "b-", label="Prediction")
plt.show()