### HOMEWORK 4: Gaussian Process for Motion Prediction ###
# Author: Marco Bueso
# Date: 11/04/2021
# Course: CS 391 Machine Learning
# Prof. Dana Ballard
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
# Convert to a 1D column from a matrix of data
def column(matrix, i):
    return [row[i] for row in matrix]

# Initialize time, and dictionaries for 5 rounds of movement by one subject 
time = []
d1_sensors = {}
d2_sensors = {}
d3_sensors = {}
d4_sensors = {}
d5_sensors = {}

#TODO: Change to data path
load_data(r"\cs391-ML-HW3-GP\data_GP\AG\block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM\20161213203046-59968-right-speed_0.500.csv", d1_sensors, num_sensors = 10, loadTime = True)
load_data(r"\cs391-ML-HW3-GP\data_GP\AG\block2-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM\20161213204004-59968-right-speed_0.500.csv", d2_sensors, num_sensors = 10)
load_data(r"\cs391-ML-HW3-GP\data_GP\AG\block3-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM\20161213204208-59968-right-speed_0.500.csv", d3_sensors, num_sensors = 10)
load_data(r"\cs391-ML-HW3-GP\data_GP\AG\block4-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM\20161213204925-59968-right-speed_0.500.csv", d4_sensors, num_sensors = 10)
load_data(r"\cs391-ML-HW3-GP\data_GP\AG\block5-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM\20161213210121-59968-right-speed_0.500.csv", d5_sensors, num_sensors = 10)



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
s_no = 5
if plot1:
    plt.plot(time,column(d2_sensors[s_no],col),time,column(d3_sensors[s_no],col),time,column(d4_sensors[s_no],col),time,column(d5_sensors[s_no],col))
    plt.show()

col_d2 = column(d2_sensors[s_no],col)
col_d3 = column(d3_sensors[s_no],col)
col_d4 = column(d4_sensors[s_no],col)
col_d5 = column(d4_sensors[s_no],col)


## GAUSSIAN PROCESS
my_kernel = C(1.0, (1e-6, 1e6)) * RBF(1030) + WhiteKernel()

# FIT 1 - 3 datasets Grouped - 2,3,4
data = []
for i in range(len(time)):
    data.append(col_d2[i])
for i in range(len(time)):
    data.append(col_d3[i])
for i in range(len(time)):
    data.append(col_d4[i])
GP = GaussianProcessRegressor(kernel = my_kernel, n_restarts_optimizer = 3)
X = np.atleast_2d(list(range(len(time)))+list(range(len(time)))+list(range(len(time)))).T # for three datasets
GP.fit(X, data)
x = np.atleast_2d(np.linspace(1, 1030, 1030)).T
y_pred, sigma = GP.predict(x, return_std=True)

#Plot
plt.figure(1)
plt.plot(X[:1030], col_d2, "r-", markersize=2, label="Observations - d2")
plt.plot(X[:1030], col_d3, "g-", markersize=2, label="Observations - d3")
plt.plot(X[:1030], col_d4, "g-", markersize=2, label="Observations - d4")
plt.plot(X[:1030], y_pred, "b-", label="Prediction")
plt.fill(
    np.concatenate([x, x[::-1]]),
    np.concatenate([y_pred - 1.9600*sigma, (y_pred + 1.9600*sigma)[::-1]]),
    alpha=0.5,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
plt.legend(loc="upper left")
plt.show()

# Second Plot - Prediction vs d5
plt.figure(2)

plt.plot(X[:1030], col_d5, "g-", markersize=2, label="Observations - d5")
plt.plot(X[:1030], y_pred, "b-", label="Prediction")
plt.fill(
    np.concatenate([x, x[::-1]]),
    np.concatenate([y_pred - 1.9600*sigma, (y_pred + 1.9600*sigma)[::-1]]),
    alpha=0.5,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
plt.legend(loc="upper left")
plt.show()


# Calculate Global Kernel Loss
# find sum of squares(d5 - y_pred)^2
sum_squares = 0
for i in range(len(col_d5)):
    sum_squares += (col_d5[i] - y_pred[i])**2
print(f"Global kernel loss: {sum_squares}")
print(f"Global kernel parameter: {GP.kernel_.get_params()['k1__k1']}")



## SLIDING WINDOW GAUSSIAN PROCESS
window = 20
delta = 10
curr = 0
prediction = []
sigmas = []
prev_prediction = np.zeros(delta)
sum_squares_local = 0
X_plot = np.atleast_2d(list(range(len(time)))).T # for three datasets
local_kernels_params = []
plt.figure(3)
while curr + window < 1030:
    data = []
    for i in range(window):
        data.append(col_d2[curr + i])
    for i in range(window):
        data.append(col_d3[curr + i])
    for i in range(window):
        data.append(col_d4[curr + i])
    GP = GaussianProcessRegressor(kernel = my_kernel, n_restarts_optimizer = 3)
    X = np.atleast_2d(list(range(window))+list(range(window))+list(range(window))).T # for three datasets
    GP.fit(X, data)
    x = np.atleast_2d(np.linspace(1, window, window)).T

    y_pred, sigma = GP.predict(x, return_std=True)
    prev_prediction_temp = y_pred
    for i in range(window - delta):
        if (prev_prediction[i] != 0):
            y_pred[i] = (y_pred[i] + prev_prediction[i + delta]) / 2
    prediction.append(y_pred[:10])
    sigmas.append(sigma[:10])
    print(f"prediction size: {len(prediction)}")
    prev_prediction = prev_prediction_temp
    # add summ squares
    for i in range(len(y_pred)):
        sum_squares_local += (col_d5[curr + i] - y_pred[i])**2

    # get kernel vals
    local_kernels_params.append(GP.kernel_.get_params()['k1__k1'])
    plt.fill(
        np.concatenate([x[:delta+1]+curr, (x[:delta+1]+curr)[::-1]]),
        np.concatenate([y_pred[:delta+1] - 1.9600*sigma[:delta+1], (y_pred[:delta+1] + 1.9600*sigma[:delta+1])[::-1]]),
        alpha=0.4,
        fc="r",
        ec="None"
    )
    # increase sliding window position
    curr += delta
prediction = np.concatenate(prediction)

#Plot
plt.plot(X_plot, col_d2, "p-", markersize=2, label="Observations - d2")
plt.plot(X_plot, col_d3, "b-", markersize=2, label="Observations - d3")
plt.plot(X_plot, col_d4, "y-", markersize=2, label="Observations - d4")
plt.plot(X_plot, col_d5, "g-", markersize=2, label="Observations - d5")
plt.plot(X_plot[:len(prediction)], prediction, "r-", label=f"Prediction")
plt.legend(loc="upper left")
plt.title("GP: Prediction vs. Observations")
plt.show()

# Calculate Local Kernel Loss
print(f"Local kernel loss: {sum_squares_local}")
print(f"Local kernel parameter: {eval(str(local_kernels_params))}")

# Plot Local Kernel Parameters
plt.figure(4)
plt.plot(eval(str(local_kernels_params)))
plt.title("Kernel parameters")
plt.show()
