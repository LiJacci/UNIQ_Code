import numpy as np

def sort_points(points, colors,n_bits):
    min_coord = np.min(points, axis=0)
    max_coord = np.max(points, axis=0)
    points_normalized = (points - min_coord) / (max_coord - min_coord + 1e-8)
    points_scaled = np.floor(points_normalized * (2**n_bits - 1)).astype(int)

    sorted_indices = np.argsort(array2vector(points_scaled,points_scaled.max()+1))

    points_sorted = points[sorted_indices]
    colors_sorted = colors[sorted_indices]

    return points_sorted, colors_sorted

def array2vector(array, step=None):
    array = np.array(array, dtype=np.int64).copy()
    step = np.array(step, dtype=np.int64).copy()
    
    if np.min(array) < 0:
        min_value = np.min(array)
        array -= min_value
        step -= min_value
    
    assert np.min(array) >= 0 and np.max(array) - np.min(array) < step
    
    vector = sum(array[:, i] * (step ** i) for i in range(array.shape[-1]))
    
    return vector

def kdtree_partition(points,feats,num_points):
    parts_geo = []
    parts_attri = []
    class KD_node:  
        def __init__(self, geo=None,attri=None, LL = None, RR = None):  
            self.geo = geo
            self.attri = attri    
            self.left = LL  
            self.right = RR
    def createKDTree(root, data_geo,data_feats):
        if len(data_geo) < num_points:
            parts_geo.append(data_geo)
            parts_attri.append(data_feats)
            return
        variances = (np.var(data_geo[:, 0]), np.var(data_geo[:, 1]), np.var(data_geo[:, 2]))
        dim_index = variances.index(max(variances))
        idx = np.lexsort(data_geo.T[dim_index, None])
        data_geo_sorted = data_geo[idx]
        data_feats_sorted = data_feats[idx]

        geo = data_geo_sorted[int(len(data_geo_sorted)/2)]
        attri = data_feats_sorted[int(len(data_feats_sorted)/2)]  
        root = KD_node(geo,attri)  
        root.left = createKDTree(root.left, data_geo_sorted[: int((len(data_geo_sorted) / 2))],data_feats_sorted[: int((len(data_feats_sorted) / 2))])  
        root.right = createKDTree(root.right, data_geo_sorted[int((len(data_geo_sorted) / 2)):],data_feats_sorted[int((len(data_feats_sorted) / 2)):]) 
        return root
    init_root = KD_node(None)
    root = createKDTree(init_root, points,feats)
    flag = False
    for i in range(len(parts_geo)):
        if not flag:
             geo_partition = np.expand_dims(parts_geo[i], axis=0)
             attri_partition = np.expand_dims(parts_attri[i], axis=0)
             flag = True
             set_num_points = parts_geo[i].shape[0]
        else:
            geo_partition = np.concatenate((geo_partition,np.expand_dims(parts_geo[i][:set_num_points,:], axis=0)),axis=0)
            attri_partition = np.concatenate((attri_partition,np.expand_dims(parts_attri[i][:set_num_points,:], axis=0)),axis=0)  
    return geo_partition,attri_partition