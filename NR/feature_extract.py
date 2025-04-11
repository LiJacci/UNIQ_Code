import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma
from pyntcloud import PyntCloud
from sklearn.neighbors import NearestNeighbors

def Entropy(labels):
    labels = labels.real
    labels = pd.to_numeric(labels, errors='coerce')
    labels = np.array(labels)
    binned_labels = pd.cut(labels, bins=2000, duplicates='drop')
    probs = binned_labels.value_counts() / len(labels)
    en = stats.entropy(probs)
    return en
    
def estimate_basic_param(vec):
    """Estimate basic parameter.
    :param vec: The vector that we want to approximate its parameter.
    :type vec: np.ndarray
    """
    result = [np.mean(vec),np.std(vec, ddof=1),Entropy(vec)]
    return result
        
def estimate_ggd_param(vec):
    """Estimate GGD parameter.
    :param vec: The vector that we want to approximate its parameter.
    :type vec: np.ndarray
    """
    gam = np.arange(0.2, 10 + 0.001, 0.001)
    r_gam = (gamma(1.0 / gam) * gamma(3.0 / gam) / (gamma(2.0 / gam) ** 2))

    sigma_sq = np.mean(vec ** 2)
    sigma = np.sqrt(sigma_sq)
    E = np.mean(np.abs(vec))
    rho = sigma_sq / (E ** 2 + 1e-10)

    alpha = gam[np.argmin(np.abs(rho-r_gam))]
    beta = sigma * np.sqrt(gamma(1/alpha)/gamma(3/alpha))
    result = [alpha,beta] ##alph=shape beta=scale sigma=variance
    return result

def gaus_norm(feats):
       from scipy.ndimage.filters import gaussian_filter
       win_sigma = 7/6
       mu  = gaussian_filter(feats, win_sigma, mode='nearest')
       sigma = np.sqrt(np.abs(gaussian_filter(np.square(feats), win_sigma, mode='nearest') - np.square(mu)))
       normalized = (feats - mu) / (sigma + 1)
       return normalized

def compute_knn_density(points,k=10):
     neigh = NearestNeighbors(n_neighbors=k)
     neigh.fit(points)
     distances, _ = neigh.kneighbors(points)
     avg_distances = np.mean(distances[:,1:], axis=1)
     avg_distances = avg_distances[avg_distances != 0]
     densities = 1 / avg_distances
     return densities.mean()

def compute_gradient_structure(points, attributes, k=100):
    attributes = attributes / 255.
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)

    gradients = []
    for i,neighbors in enumerate(indices):
        center_p = attributes[i]
        neighbors_p = attributes[neighbors]
        diff = neighbors_p - center_p
        gradient_magnitude = np.linalg.norm(diff,axis=1)
          # 计算邻域的属性梯度
        gradients.append(np.mean(gradient_magnitude))
    gradients = np.array(gradients)

    return gradients

def get_color_nss_param(vec):
    return [estimate_basic_param(vec),estimate_ggd_param(vec)]

def get_geometry_nss_param(vec):
    return [estimate_basic_param(vec),estimate_ggd_param(vec)]

def get_feature_vector_data(coord, feats):
    points_data = np.hstack((coord, feats))
    cloud = PyntCloud(pd.DataFrame(points_data, columns=['x', 'y', 'z', 'r', 'g', 'b'][:points_data.shape[1]]))
    k = min(coord.shape[0]-1, 100)
    k_neighbors = cloud.get_neighbors(k=k)
    
    ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
    
    cloud.add_scalar_field("linearity",ev=ev)
    linearity = cloud.points[f'linearity({k+1})'].to_numpy()

    cloud.add_scalar_field("planarity",ev=ev)
    planarity = cloud.points[f'planarity({k+1})'].to_numpy()

    cloud.add_scalar_field("sphericity",ev=ev)
    sphericity = cloud.points[f'sphericity({k+1})'].to_numpy()

    cloud.add_scalar_field("curvature", ev=ev)
    curvature = cloud.points[f'curvature({k + 1})'].to_numpy()

    #begin color 
    channels = [feats[:,0],feats[:,1],feats[:,2]]
    gradient = compute_gradient_structure(coord, feats)

    ##compute nss parameters
    color_params = []
    geo_params = []
    # compute color nss features
    for tmp in [gradient] + channels:
        tmp = gaus_norm(tmp)
        params = get_color_nss_param(tmp)
        flat_params = [i for item in params for i in item]
        color_params = color_params + flat_params

    # compute geomerty nss features
    for tmp in [curvature,linearity,planarity,sphericity]:
        tmp = gaus_norm(tmp)
        params = get_geometry_nss_param(tmp)
        flat_params = [i for item in params for i in item]
        geo_params = geo_params + flat_params

    return color_params,geo_params