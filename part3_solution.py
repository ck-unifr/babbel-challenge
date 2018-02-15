# Part 2 - Advanced Python
# author: Kai Chen
# date: Jan. 2017

from sklearn import datasets

from sklearn.mixture import GMM
from sklearn.neighbors import KernelDensity

from scipy.stats import kde
import matplotlib.pyplot as plt

import numpy as np

# Load data from sklearn
D, _ = datasets.make_classification( n_samples=200,
                                    n_features=2,
                                    n_informative=2,
                                    n_redundant=0,
                                    n_classes=3,
                                    n_clusters_per_class=1,
                                    class_sep=2 )

x, y = D[:,0], D[:,1]
plt.scatter(x, y)
plt.show()


# Display the histograms to get some ideas about the density of the given data
nbins=50
plt.hexbin(x, y, gridsize=nbins)
plt.show()

plt.hist2d(x, y, bins=nbins)
plt.show()


# 1.a Perform density estimation using Kernel Density Estimator (KDE) with Gaussian kernel

# Create a KED model with scipy
"""Note:
Bandwidth selection strongly influences the estimate obtained from the KDE.
scipy includes automatic bandwidth determination.
The estimation works best for a unimodal distribution; bimodal or multi-modal distributions tend to be oversmoothed.
For the details of bandwidth selection in scipy, we refer to:
https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.stats.gaussian_kde.html
That's why the heatmap of the KDE created with scipy makes more sense than the heatmap of the KDE created with sklearn.
"""
k = kde.gaussian_kde(D.T)
X, Y = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
Z = k(np.vstack([X.flatten(), Y.flatten()]))
plt.pcolormesh(X, Y, Z.reshape(X.shape))
plt.title("KED (scipy)")
plt.show()
# show band width
#print(k.factor)


# Create a KED model With sklearn
kde = KernelDensity(kernel='gaussian', bandwidth=0.6).fit(D)
X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 200), np.linspace(y.min(), y.max(), 200))
XX = np.array([X.ravel(), Y.ravel()]).T
Z = kde.score_samples(XX)
plt.pcolormesh(X, Y, Z.reshape(X.shape))
plt.title("KED (sklearn)")
plt.show()



# 1.b Perform density estimation using Gaussian Mixture Model (GMM).

# Create a GMM model with sklearn
gmm = GMM(n_components=3, covariance_type='full')
gmm.fit(D)
#X, Y = np.meshgrid(np.linspace(x.min(), x.max()), np.linspace(y.min(), y.max()))
# #XX = np.array([X.ravel(), Y.ravel()]).T
Z, _ = gmm.score_samples(XX)
plt.pcolormesh(X, Y, Z.reshape(X.shape))
plt.title("GMM (sklearn)")
plt.show()



# 2 Approximate P(X) and P(Y)

# Scale the data, such that d > 0, for all d \in D
#D[:,0] = D[:,0] + abs(D[:,0].min())
#D[:,1] = D[:,1] + abs(D[:,1].min())
#x, y = D[:,0], D[:,1]

nb_samples = 500
x_plot = np.linspace(x.min(), x.max(), nb_samples)
y_plot = np.linspace(y.min(), y.max(), nb_samples)

# First step: project all samples into the x-axis
# bin_dist = 0.5
# histX = np.zeros(int(x.max()) - int(x.min()) + 1)
# for bin_x in range(int(x.min()), int(x.max())):
#   for d in D:
#      if (abs(bin_x - d[0]) < bin_dist):
#         histX[bin_x] = histX[bin_x] + 1

# Fit the data x with a KDE
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x.reshape(-1, 1))
px = kde.score_samples(x_plot.reshape(-1, 1))
plt.plot(x_plot, px)
plt.title("P(X) KDE #samples=%d" % (nb_samples))
plt.show()

# Fit the data x with a GMM
gmm = GMM(n_components=3, covariance_type='full')
gmm.fit(x.reshape(-1, 1))
px, _ = gmm.score_samples(x_plot.reshape(-1, 1))
plt.plot(x_plot, px)
plt.title("P(X) GMM #samples=%d" % (nb_samples))
plt.show()

# Fit the data y with a KDE
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(y.reshape(-1, 1))
py = kde.score_samples(y_plot.reshape(-1, 1))
plt.plot(y_plot, py)
plt.title("P(y) KDE #samples=%d" % (nb_samples))
plt.show()


# Fit the data y with a GMM
gmm = GMM(n_components=3, covariance_type='full')
gmm.fit(y.reshape(-1, 1))
py, _ = gmm.score_samples(y_plot.reshape(-1, 1))
plt.plot(y_plot, py)
plt.title("P(y) GMM #samples=%d" % (nb_samples))
plt.show()


# 3 Bandwidth selection

nb_samples = 200
X, Y = np.meshgrid(np.linspace(x.min(), x.max(), nb_samples), np.linspace(y.min(), y.max(), nb_samples))
XX = np.array([X.ravel(), Y.ravel()]).T


def get_diff_l2(score1, score2):
   diff = [(score1[i] - score2[i])**2 for i in range(len(score1))]
   return sum(diff)/len(score1)

def get_optimal_bandwidth_lin(D, scores_groundtruth, epochs=10, initial_bandwidth=1, step_size=-0.1):
   bandwidth = initial_bandwidth
   errors = []

   previous_error = float("inf")
   for i in range(epochs):
      kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(D)
      scores_kde = kde.score_samples(XX)
      error = get_diff_l2(scores_groundtruth, scores_kde)

      print("epochs = %d, error = %f" % (i, error))
      errors.append(error)

      # stop loop when error does not decrease too much
      if(error >= previous_error):
         del errors[-1]
         bandwidth = bandwidth - step_size
         return bandwidth, errors, i

      bandwidth = bandwidth + step_size
      previous_error = error

   return bandwidth, errors, epochs


epochs=10
initial_bandwidth=1
step_size=-0.1

# Compute scores of GMM and consider these scores as our ground truth scores
gmm = GMM(n_components=3, covariance_type='full')
gmm.fit(D)
scores_gmm, _ = gmm.score_samples(XX)

bandwidth, errors, nb_iterations = get_optimal_bandwidth_lin(D,
                                                         scores_groundtruth=scores_gmm,
                                                         epochs=epochs,
                                                         initial_bandwidth=initial_bandwidth,
                                                         step_size=step_size)
plt.plot(range(nb_iterations), errors)
plt.show()


# Fit the data x with a KED
kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x.reshape(-1, 1))
px = kde.score_samples(x_plot.reshape(-1, 1))
plt.plot(x_plot, px)
plt.title("P(X) KDE #samples=%d" % (nb_samples))
plt.show()

# Fit the data y with a KED
kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(y.reshape(-1, 1))
py = kde.score_samples(y_plot.reshape(-1, 1))
plt.plot(y_plot, py)
plt.title("P(y) KDE #samples=%d" % (nb_samples))
plt.show()



def get_optimal_bandwidth_bin(D, scores_groundtruth, initial_bandwidth=2, smallest_bandwidth=0.1, episilon=0.1):

   # The list contains three bandwidth values, i.e., left, right, and middle
   bandwidth = [initial_bandwidth, (initial_bandwidth+smallest_bandwidth)/2, smallest_bandwidth]

   # errors obtain with the three bandwidth values
   errors = [0, 0, 0]

   # store the error value of the middle bandwidth
   optimal_errors = []

   nb_iteration = 0

   kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth[0]).fit(D)
   errors[0] = get_diff_l2(scores_groundtruth, kde.score_samples(XX))
   kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth[2]).fit(D)
   errors[2] = get_diff_l2(scores_groundtruth, kde.score_samples(XX))

   while True:

      kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth[1]).fit(D)
      errors[1] = get_diff_l2(scores_groundtruth, kde.score_samples(XX))

      nb_iteration += 1
      optimal_errors.append(errors[1])
      print("epochs = %d, error = %f" % (nb_iteration, errors[1]))

      # start new search between the two bandwidth where we achieve two minimum errors
      bandwidth = [b for (e, b) in sorted(zip(errors, bandwidth))]
      bandwidth[2] = bandwidth[1]
      bandwidth[1] = (bandwidth[0] + bandwidth[1])/2
      sorted(errors)

      if(abs(bandwidth[0]-bandwidth[1]) < episilon):
         return bandwidth[1], optimal_errors, nb_iteration

   return bandwidth[1], optimal_errors, nb_iteration


bandwidth, errors, nb_iterations = get_optimal_bandwidth_bin(D,
                                                             scores_groundtruth=scores_gmm,
                                                             initial_bandwidth=1,
                                                             smallest_bandwidth=0.1)
plt.plot(range(nb_iterations), errors)
plt.show()

# Fit the data x with a KED
kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x.reshape(-1, 1))
px = kde.score_samples(x_plot.reshape(-1, 1))
plt.plot(x_plot, px)
plt.title("P(X) KDE #samples=%d" % (nb_samples))
plt.show()

# Fit the data y with a KED
kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(y.reshape(-1, 1))
py = kde.score_samples(y_plot.reshape(-1, 1))
plt.plot(y_plot, py)
plt.title("P(y) KDE #samples=%d" % (nb_samples))
plt.show()
