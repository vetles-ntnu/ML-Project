# SHOULD BE RUN WITH THE EDITED FILE "WDBC_features_visualization.csv"

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv("WDBC_features_visualization.csv")
data.head()

radius_mean=data['radius_mean']
texture_mean=data['texture_mean']
perimeter_mean=data['perimeter_mean']
area_mean=data['area_mean']
smoothness_mean=data['smoothness_mean']
compactness_mean=data['compactness_mean']
concavity_mean=data['concavity_mean']
concave_points_mean=data['concave points_mean']
symmetry_mean=data['symmetry_mean']
fractal_dimension_mean=data['fractal_dimension_mean']

# benign samples 212: rows
radius_mean_b = radius_mean[212:]
texture_mean_b = texture_mean[212:]
perimeter_mean_b = perimeter_mean[212:]
area_mean_b = area_mean[212:]
smoothness_mean_b = smoothness_mean[212:]
compactness_mean_b = compactness_mean[212:]
concavity_mean_b = concavity_mean[212:]
concave_points_mean_b = concave_points_mean[212:]
symmetry_points_mean_b = symmetry_mean[212:]
fractal_dimension_mean_b = fractal_dimension_mean[212:]

# malign samples 1:212 rows
radius_mean_m = radius_mean[0:212]
texture_mean_m = texture_mean[0:212]
perimeter_mean_m = perimeter_mean[0:212]
area_mean_m = area_mean[0:212]
smoothness_mean_m = smoothness_mean[0:212]
compactness_mean_m = compactness_mean[0:212]
concavity_mean_m = concavity_mean[0:212]
concave_points_mean_m = concave_points_mean[0:212]
symmetry_points_mean_m = symmetry_mean[0:212]
fractal_dimension_mean_m = fractal_dimension_mean[0:212]

o = np.ones(len(area_mean_m))
z = np.zeros(len(area_mean_b))

fig, axs = plt.subplots(5, 2, figsize=(6,6), constrained_layout=True)
plt.setp(axs, yticks=[0, 1], yticklabels=['B', 'M'])
axs[0, 0].scatter(radius_mean_m, o, c='Red', marker='.')
axs[0, 0].scatter(radius_mean_b, z, c='Green', marker='.')
axs[0, 0].set_xlabel('radius mean', fontsize=12)

axs[0, 1].scatter(texture_mean_m, o, c='Red', marker='.')
axs[0, 1].scatter(texture_mean_b, z, c='Green', marker='.')
axs[0, 1].set_xlabel('texture mean', fontsize=12)

axs[1, 0].scatter(perimeter_mean_m, o, c='Red', marker='.')
axs[1, 0].scatter(perimeter_mean_b, z, c='Green', marker='.')
axs[1, 0].set_xlabel('perimeter mean', fontsize=12)

axs[1, 1].scatter(area_mean_m, o, c='Red', marker='.')
axs[1, 1].scatter(area_mean_b, z, c='Green', marker='.')
axs[1, 1].set_xlabel('area mean', fontsize=12)

axs[2, 0].scatter(smoothness_mean_m, o, c='Red', marker='.')
axs[2, 0].scatter(smoothness_mean_b, z, c='Green', marker='.')
axs[2, 0].set_xlabel('smoothness mean', fontsize=12)

axs[2, 1].scatter(compactness_mean_m, o, c='Red', marker='.')
axs[2, 1].scatter(compactness_mean_b, z, c='Green', marker='.')
axs[2, 1].set_xlabel('compactness mean', fontsize=12)

axs[3, 0].scatter(concavity_mean_m, o, c='Red', marker='.')
axs[3, 0].scatter(concavity_mean_b, z, c='Green', marker='.')
axs[3, 0].set_xlabel('concavity mean', fontsize=12)

axs[3, 1].scatter(concave_points_mean_m, o, c='Red', marker='.')
axs[3, 1].scatter(concave_points_mean_b, z, c='Green', marker='.')
axs[3, 1].set_xlabel('concave points mean', fontsize=12)

axs[4, 0].scatter(symmetry_points_mean_m, o, c='Red', marker='.')
axs[4, 0].scatter(symmetry_points_mean_b, z, c='Green', marker='.')
axs[4, 0].set_xlabel('symmetry mean', fontsize=12)

axs[4, 1].scatter(fractal_dimension_mean_m, o, c='Red', marker='.')
axs[4, 1].scatter(fractal_dimension_mean_b, z, c='Green', marker='.')
axs[4, 1].set_xlabel('fractal dimension mean', fontsize=12)

#plt.savefig('features_mean', dpi=300)

plt.show()

radius_se=data['radius_se']
texture_se=data['texture_se']
perimeter_se=data['perimeter_se']
area_se=data['area_se']
smoothness_se=data['smoothness_se']
compactness_se=data['compactness_se']
concavity_se=data['concavity_se']
concave_points_se=data['concave points_se']
symmetry_se=data['symmetry_se']
fractal_dimension_se=data['fractal_dimension_se']

radius_se_b = radius_se[212:]
texture_se_b = texture_se[212:]
perimeter_se_b = perimeter_se[212:]
area_se_b = area_se[212:]
smoothness_se_b = smoothness_se[212:]
compactness_se_b = compactness_se[212:]
concavity_se_b = concavity_se[212:]
concave_points_se_b = concave_points_se[212:]
symmetry_points_se_b = symmetry_se[212:]
fractal_dimension_se_b = fractal_dimension_se[212:]

radius_se_m = radius_se[0:212]
texture_se_m = texture_se[0:212]
perimeter_se_m = perimeter_se[0:212]
area_se_m = area_se[0:212]
smoothness_se_m = smoothness_se[0:212]
compactness_se_m = compactness_se[0:212]
concavity_se_m = concavity_se[0:212]
concave_points_se_m = concave_points_se[0:212]
symmetry_points_se_m = symmetry_se[0:212]
fractal_dimension_se_m = fractal_dimension_se[0:212]

fig, axs = plt.subplots(5, 2, figsize=(6,6), constrained_layout=True)
plt.setp(axs, yticks=[0, 1], yticklabels=['B', 'M'])
axs[0, 0].scatter(radius_se_m, o, c='Red', marker='.')
axs[0, 0].scatter(radius_se_b, z, c='Green', marker='.')
axs[0, 0].set_xlabel('radius SE', fontsize=12)

axs[0, 1].scatter(texture_se_m, o, c='Red', marker='.')
axs[0, 1].scatter(texture_se_b, z, c='Green', marker='.')
axs[0, 1].set_xlabel('texture SE', fontsize=12)

axs[1, 0].scatter(perimeter_se_m, o, c='Red', marker='.')
axs[1, 0].scatter(perimeter_se_b, z, c='Green', marker='.')
axs[1, 0].set_xlabel('perimeter SE', fontsize=12)

axs[1, 1].scatter(area_se_m, o, c='Red', marker='.')
axs[1, 1].scatter(area_se_b, z, c='Green', marker='.')
axs[1, 1].set_xlabel('area SE', fontsize=12)

axs[2, 0].scatter(smoothness_se_m, o, c='Red', marker='.')
axs[2, 0].scatter(smoothness_se_b, z, c='Green', marker='.')
axs[2, 0].set_xlabel('smoothness SE', fontsize=12)

axs[2, 1].scatter(compactness_se_m, o, c='Red', marker='.')
axs[2, 1].scatter(compactness_se_b, z, c='Green', marker='.')
axs[2, 1].set_xlabel('compactness SE', fontsize=12)

axs[3, 0].scatter(concavity_se_m, o, c='Red', marker='.')
axs[3, 0].scatter(concavity_se_b, z, c='Green', marker='.')
axs[3, 0].set_xlabel('concavity SE', fontsize=12)

axs[3, 1].scatter(concave_points_se_m, o, c='Red', marker='.')
axs[3, 1].scatter(concave_points_se_b, z, c='Green', marker='.')
axs[3, 1].set_xlabel('concave points SE', fontsize=12)

axs[4, 0].scatter(symmetry_points_se_m, o, c='Red', marker='.')
axs[4, 0].scatter(symmetry_points_se_b, z, c='Green', marker='.')
axs[4, 0].set_xlabel('symmetry SE', fontsize=12)

axs[4, 1].scatter(fractal_dimension_se_m, o, c='Red', marker='.')
axs[4, 1].scatter(fractal_dimension_se_b, z, c='Green', marker='.')
axs[4, 1].set_xlabel('fractal dimension SE', fontsize=12)

#plt.savefig('features_se', dpi=300)

plt.show()

radius_w=data['radius_worst']
texture_w=data['texture_worst']
perimeter_w=data['perimeter_worst']
area_w=data['area_worst']
smoothness_w=data['smoothness_worst']
compactness_w=data['compactness_worst']
concavity_w=data['concavity_worst']
concave_points_w=data['concave points_worst']
symmetry_w=data['symmetry_worst']
fractal_dimension_w=data['fractal_dimension_worst']

radius_w_b = radius_w[212:]
texture_w_b = texture_w[212:]
perimeter_w_b = perimeter_w[212:]
area_w_b = area_w[212:]
smoothness_w_b = smoothness_w[212:]
compactness_w_b = compactness_w[212:]
concavity_w_b = concavity_w[212:]
concave_points_w_b = concave_points_w[212:]
symmetry_points_w_b = symmetry_w[212:]
fractal_dimension_w_b = fractal_dimension_w[212:]

radius_w_m = radius_w[0:212]
texture_w_m = texture_w[0:212]
perimeter_w_m = perimeter_w[0:212]
area_w_m = area_w[0:212]
smoothness_w_m = smoothness_w[0:212]
compactness_w_m = compactness_w[0:212]
concavity_w_m = concavity_w[0:212]
concave_points_w_m = concave_points_w[0:212]
symmetry_points_w_m = symmetry_w[0:212]
fractal_dimension_w_m = fractal_dimension_w[0:212]

fig, axs = plt.subplots(5, 2, figsize=(6,6), constrained_layout=True)
plt.setp(axs, yticks=[0, 1], yticklabels=['B', 'M'])
axs[0, 0].scatter(radius_w_m, o, c='Red', marker='.')
axs[0, 0].scatter(radius_w_b, z, c='Green', marker='.')
axs[0, 0].set_xlabel('radius "worst"', fontsize=12)

axs[0, 1].scatter(texture_w_m, o, c='Red', marker='.')
axs[0, 1].scatter(texture_w_b, z, c='Green', marker='.')
axs[0, 1].set_xlabel('texture "worst"', fontsize=12)

axs[1, 0].scatter(perimeter_w_m, o, c='Red', marker='.')
axs[1, 0].scatter(perimeter_w_b, z, c='Green', marker='.')
axs[1, 0].set_xlabel('perimeter "worst"', fontsize=12)

axs[1, 1].scatter(area_w_m, o, c='Red', marker='.')
axs[1, 1].scatter(area_w_b, z, c='Green', marker='.')
axs[1, 1].set_xlabel('area "worst"', fontsize=12)

axs[2, 0].scatter(smoothness_w_m, o, c='Red', marker='.')
axs[2, 0].scatter(smoothness_w_b, z, c='Green', marker='.')
axs[2, 0].set_xlabel('smoothness "worst"', fontsize=12)

axs[2, 1].scatter(compactness_w_m, o, c='Red', marker='.')
axs[2, 1].scatter(compactness_w_b, z, c='Green', marker='.')
axs[2, 1].set_xlabel('compactness "worst"', fontsize=12)

axs[3, 0].scatter(concavity_w_m, o, c='Red', marker='.')
axs[3, 0].scatter(concavity_w_b, z, c='Green', marker='.')
axs[3, 0].set_xlabel('concavity "worst"', fontsize=12)

axs[3, 1].scatter(concave_points_w_m, o, c='Red', marker='.')
axs[3, 1].scatter(concave_points_w_b, z, c='Green', marker='.')
axs[3, 1].set_xlabel('concave points "worst"', fontsize=12)

axs[4, 0].scatter(symmetry_points_w_m, o, c='Red', marker='.')
axs[4, 0].scatter(symmetry_points_w_b, z, c='Green', marker='.')
axs[4, 0].set_xlabel('symmetry "worst"', fontsize=12)

axs[4, 1].scatter(fractal_dimension_w_m, o, c='Red', marker='.')
axs[4, 1].scatter(fractal_dimension_w_b, z, c='Green', marker='.')
axs[4, 1].set_xlabel('fractal dimension "worst"', fontsize=12)

#plt.savefig('features_w', dpi=300)

plt.show()
