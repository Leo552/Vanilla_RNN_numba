import numpy as np
import numba_neural_network as nn
import z_helper as h
import time

import pandas as pd


# ----------------------------- Compile the NN ----------------------------- #

# Toy data used to compile the neural network
d = np.array([[[1.0]]])
s = 1

print('Compiling...')
begin_time = time.time_ns()
compiled_nn_values = nn.make_neural_network(layer_sizes=[s,s], layer_activations=[h.softmax])
nn.train_auto(d,d,d,d, compiled_nn_values)
end_time = time.time_ns()
print("Compile time:", (end_time-begin_time) / 1e9)


# ----------------------------- Import the data ----------------------------- #

print('Reading data from csv... ', end = '')
FILENAME_PATH = 'C:\\Users\\maxel\\OneDrive\\Machine_learning\\Stock forecaster\\Vanilla_RNN_numba\\Datasets\\'
df = pd.read_csv(FILENAME_PATH + "EURUSD_scaled_small.csv")

data_input = df[['HIGH','LOW','OPEN','CLOSE']][:-1].to_numpy()
data_output = df[['HIGH','LOW','OPEN','CLOSE']][1:].to_numpy()

# Adjust the dimensions
data_input = np.expand_dims(data_input, axis = 2)
data_output = np.expand_dims(data_output, axis = 2)
print('Done')


# ------------------------------ Simple train ------------------------------ #

# Split the data - 60% train, 20% validate, 20% test
np.random.seed(420)
train_input, validate_input, test_input = h.kfold(5, data_input)
train_output, validate_output, test_output = h.kfold(5, data_output)
print('Data split')

nn_values = nn.make_neural_network(layer_sizes=[train_input.shape[1], 20, train_output.shape[1]],
                                   layer_activations=[h.sigmoid, h.identity])
print('Values types compiled')
begin_time = time.time_ns()
epochs, current_mse = nn.train_auto(train_input, train_output, validate_input, 
                                    validate_output, nn_values, 1000)
end_time = time.time_ns()

train_mse = nn.calculate_MSE(train_input, train_output, nn_values)
test_mse = nn.calculate_MSE(test_input, test_output, nn_values)

print("Epochs:", epochs, 
      "Time:", (end_time-begin_time)/1e9, 
      "Tr:", train_mse,
      "V:", current_mse, 
      "T:", test_mse)


# ----------------------------- Evaluation loop ----------------------------- #

np.random.seed(420)
total_accuracy = 0.0
begin_total = time.time_ns()
n = 10
for i in range(n):

    random_seed = np.random.randint(10, 1010)
    np.random.seed(random_seed)

    train_input, validate_input, test_input = h.kfold(7, data_input, random_seed)
    train_output, validate_output, test_output = h.kfold(7, data_output, random_seed)

    nn_values = nn.make_neural_network(layer_sizes=[train_input.shape[1], 20, train_output.shape[1]], layer_activations=[h.sigmoid, h.softmax])

    begin_time = time.time_ns()
    epochs, current_mse = nn.train_auto(train_input, train_output, validate_input, validate_output, nn_values)
    end_time = time.time_ns()

    train_mse = nn.calculate_MSE(train_input, train_output, nn_values)
    test_mse = nn.calculate_MSE(test_input, test_output, nn_values)

    accuracy_test = nn.evaluate(test_input, test_output, nn_values)
    total_accuracy += accuracy_test
    print("Seed:", random_seed, "Epochs:", epochs, "Time:", (end_time-begin_time)/1e9, "Accuracy:", accuracy_test, "Tr:", train_mse, "V:", current_mse, "T:", test_mse)
print("Average Accuracy:", total_accuracy / n, "Average Time:", ((time.time_ns()-begin_total)/1e9) / n)
