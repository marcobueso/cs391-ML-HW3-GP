
import csv
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

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
# num_side = 2
# fig, axs = plt.subplots(num_side, num_side)
# for i in range(num_side):
#     for j in range(num_side):
#         print(i,j)
#         axs[i,j].plot(time, column(d1_sensors[i+j],1))
#         axs[i,j].axis('off')
# plt.show()


### PLOT sensor s_no's 'y' component data, for 4 trials, dictionary 2-5
col = 1
s_no = 6
if plot1:
    plt.plot(time,column(d2_sensors[s_no],col),time,column(d3_sensors[s_no],col),time,column(d4_sensors[s_no],col),time,column(d5_sensors[s_no],col))
    plt.show()


## GAUSSIAN PROCESS
my_kernel = C(1.0) * RBF(1030) + WhiteKernel()
GP = GaussianProcessRegressor(kernel = my_kernel)#TODO: changed: , n_restarts_optimizer = 4)

X = np.atleast_2d(range(len(time))).T
col_d2 = column(d2_sensors[s_no],col)
print(len(col_d2))
col_d3 = column(d3_sensors[s_no],col)
d = {'d2': col_d2, 'd3': col_d3}
df = pd.DataFrame(data=d)
GP.fit(X, df)
GP.get_params()
x = np.atleast_2d(np.linspace(1, 1030, 1030)).T

y_pred, sigma = GP.predict(x, return_std=True)
# plt.figure()
# plt.plot(X, column(d2_sensors[s_no],col), "r-", markersize=2, label="Observations")
# plt.plot(X, y_pred, "b-", label="Prediction")
# plt.show()
data = np.atleast_2d(range(len(time)))
print(data)
#for i in range(len(time)):
    



fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(X, column(d2_sensors[s_no],col), "r-", markersize=2, label="Observations")
axs[0, 0].set_title('Axis [0, 0]: Observations')
axs[0, 1].plot(X, y_pred, "b-", label="Prediction")
axs[0, 1].plot(X, column(d2_sensors[s_no],col), "r-", markersize=2, label="Observations")
axs[0, 1].plot(X, col_d3, "g-", markersize=2)
axs[0,1].fill(
    np.concatenate([x, x[::-1]]),
    #np.concatenate([y_pred - 1.9600*sigma, (y_pred + 1.9600*sigma)[::-1]]),
    np.concatenate([column(y_pred,1) - 1.9600*sigma, (column(y_pred,1) + 1.9600*sigma)[::-1]]),
    alpha=0.5,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
axs[0,1].legend(loc="upper left")
axs[0, 1].set_title('Axis [0, 1]: Prediction')

# FIT WITH DICT 3
GP.fit(X,column(d3_sensors[s_no],col))
x = np.atleast_2d(np.linspace(1, 1030, 1030)).T

y_pred, sigma = GP.predict(x, return_std=True)

axs[1, 0].plot(X, y_pred, "g-", label="Prediction")
axs[1, 0].plot(X, column(d3_sensors[s_no],col), "r-", markersize=2, label="Observations")
axs[1, 0].fill(
    np.concatenate([x, x[::-1]]),
    np.concatenate([y_pred - 1.9600*sigma, (y_pred + 1.9600*sigma)[::-1]]),
    alpha=0.5,
    fc="g",
    ec="None",
    label="95% confidence interval",
)
axs[1, 0].legend(loc="upper left")
axs[1, 0].set_title('Axis [1 0]: Prediction after Dict 3')

#axs[1, 0].plot(X, column(d2_sensors[s_no],col), "r-", markersize=2, label="Observations")
#axs[1, 0].plot(X, y_pred, "b-", label="Prediction")
#axs[1, 0].set_title('Both')
# axs[1, 1].plot(x, column(d2_sensors[s_no],col), 'tab:red')
# axs[1, 1].set_title('Ignore')
plt.show()