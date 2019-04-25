import lib
import numpy as np

import sklearn.linear_model as lm
import matplotlib.pyplot as plt

# Model = linear regression
model = lm.LinearRegression()

# Fit models on dataset
n_features, r2_train, r2_test, snr = lib.fit_on_increasing_size(model)
argmax = n_features[np.argmax(r2_test)]

# plot
fig, axis = plt.subplots(1, 2, figsize=(9, 3))

# Left pane: all features
lib.plot_r2_snr(n_features, r2_train, r2_test, argmax, snr, axis[0])

# Right pane: Zoom on 100 first features
lib.plot_r2_snr(
    n_features[n_features <= 100],
    r2_train[n_features <= 100],
    r2_test[n_features <= 100],
    argmax,
    snr[n_features <= 100],axis[1]
)
plt.tight_layout()
plt.show()