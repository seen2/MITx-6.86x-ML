import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


train_x, train_y, test_x, test_y = get_MNIST_data()
# TODO: First fill out cubicFeatures() function in features.py as the below code requires it.
n_components = 10

###Correction note:  the following 4 lines have been modified since release.
train_x_centered, feature_means = center_data(train_x)
pcs = principal_components(train_x_centered)
train_pca10 = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca10 = project_onto_PC(test_x, pcs, n_components, feature_means)
train_cube = cubic_features(train_pca10)
test_cube = cubic_features(test_pca10)
# train_cube (and test_cube) is a representation of our training (and test) data
# after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.

train_x, train_y, test_x, test_y = get_MNIST_data()
# TODO: First fill out cubicFeatures() function in features.py as the below code requires it.
n_components = 10

###Correction note:  the following 4 lines have been modified since release.
train_x_centered, feature_means = center_data(train_x)
pcs = principal_components(train_x_centered)
train_pca10 = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca10 = project_onto_PC(test_x, pcs, n_components, feature_means)
train_cube = cubic_features(train_pca10)
test_cube = cubic_features(test_pca10)
# train_cube (and test_cube) is a representation of our training (and test) data
# after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.


# TODO: Train your softmax regression model using (train_cube, train_y)
#       and evaluate its accuracy on (test_cube, test_y).
pol_model=SVC(kernel = 'rbf')
pol_model.fit(train_cube, train_y)
test_error = 1 - np.mean(pol_model.predict(test_cube) == test_y)
#Save the model parameters theta obtained from calling softmax_regression to disk.
# write_pickle_data(theta, "./theta.pkl.gz")
print(">>>>>>>>>>>>>>>>>>>>PCA 10 Error>>>>>>>>>>>>>>>>>>>>>>>")
print('Error=', test_error)