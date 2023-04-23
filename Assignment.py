# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 12:01:45 2023

@author: Salman Saleem Khan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_name):
 
    # Load data from a CSV file, skip any bad lines.
    data = pd.read_csv(file_name)
    return data.to_numpy()


def compute_newborn_weight_distribution(file_name, num_bins=10):
    # Load the data from the CSV file
    weights = np.loadtxt(file_name, delimiter=',')

    # Compute the histogram of weights
    hist, bins = np.histogram(weights, bins=num_bins)

    # Calculate the average weight
    w_tilde = np.average(weights)

    # Calculate the value of X such that 75% of newborns have a weight above X
    X = np.percentile(weights, q=25)

    # Plot the histogram and add the values of W˜ and X to the title
    plt.hist(weights, bins=num_bins, color='blue', alpha=0.5)
    plt.xlabel('Newborn Weight (Kilograms)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Newborn Weights\nW˜ = {:.2f} Kilograms, X = {:.2f} Kilograms'.format(w_tilde, X))

    # Return the histogram, bin edges, average weight, and value of X
    return hist, bins, w_tilde, X


hist, bins, w_tilde, X = compute_newborn_weight_distribution('data0.csv')
plt.show()
print("Average weight of newborns: {:.2f} Kilograms".format(w_tilde))
print("Value of X such that 75% of newborns are above X: {:.2f} Kilograms".format(X))


