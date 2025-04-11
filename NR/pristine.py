from tqdm import tqdm
import argparse, os
import numpy as np
from pyntcloud import PyntCloud
from feature_extract import get_feature_vector_data
from utils import sort_points, kdtree_partition

def load_data(args, filedir):
    cloud = PyntCloud.from_file(filedir)
    coord = np.array(cloud.points)[:,0:3]
    if args.dataset == 'sjtu':
        feats = np.array(cloud.points)[:,6:9]
    else:
        feats = np.array(cloud.points)[:,3:6]
    return coord, feats

def estimatemodelpara(args):   
### load dataset
    file_paths = os.listdir(args.filedirs)
    file_paths = sorted(file_paths)
    print('dbg:', len(file_paths))
    color_params_list, geo_params_list = [],[]
    for i,file in enumerate(tqdm(file_paths)):
        index = file.find(".", 0) 
        filename = file[0:index] 
        print('load point cloud', filename)
        filedir = os.path.join(args.filedirs, file)
        coord, feats = load_data(args,filedir) 
        coord, feats = sort_points(coord, feats, n_bits=10)

        ## block partition
        coord_block, feats_block = kdtree_partition(coord,feats,num_points=args.K)
        coord_block = [coord_block[p,:,:] for p in range(coord_block.shape[0])]
        feats_block = [feats_block[q,:,:] for q in range(feats_block.shape[0])]

        ### compute features of each block
        from joblib import Parallel, delayed
        def process_patch(j):
            coord_j = coord_block[j].reshape(-1,3)
            feats_j = feats_block[j].reshape(-1,3)
            color_params,geo_params = get_feature_vector_data(coord_j,feats_j)
            color = np.expand_dims(color_params,1) 
            geo = np.expand_dims(geo_params,1)
            return color, geo
        results = Parallel(n_jobs=-1)(delayed(process_patch)(j) for j in range(len(coord_block)))
        color_one = np.concatenate([res[0] for res in results], axis=1) #(feature_num, patches_n)
        geo_one = np.concatenate([res[1] for res in results], axis=1)

        color_params_list.append(color_one)
        geo_params_list.append(geo_one)
    color_params_array = np.concatenate(color_params_list, axis=1) #(feature_num, patches_N)
    geo_params_array = np.concatenate(geo_params_list, axis=1)

### fit gaussian distribution for patches
    #mean of color and all geo
    def meanofdata(paramsarray):
        meanofdata = np.mean(paramsarray, axis=1,keepdims=True)
        meanMatric = np.tile(meanofdata,(1,paramsarray.shape[1]))
        centerlized = paramsarray - meanMatric
        return centerlized,meanofdata
    color_params_array,color_mean = meanofdata(color_params_array)
    geo_params_array,geo_mean = meanofdata(geo_params_array)
    
    def cross_block(color_params_array,geo_params_array):
        color_patch_mean = np.nanmean(color_params_array, axis=1) #(feature_num,)
        geo_patch_mean = np.nanmean(geo_params_array, axis=1) #(feature_num,)
        distparam_no_nan_color = color_params_array[:,~np.isnan(color_params_array).any(axis=0)]
        color_patch_std = np.cov(distparam_no_nan_color, rowvar=True)
        distparam_no_nan_geo = geo_params_array[:,~np.isnan(geo_params_array).any(axis=0)]
        geo_patch_std = np.cov(distparam_no_nan_geo, rowvar=True)
        return color_patch_mean,geo_patch_mean,color_patch_std,geo_patch_std
    color_patch_mean,geo_patch_mean,color_patch_std,geo_patch_std = cross_block(color_params_array,geo_params_array)
    

### get and save modelpara
    para_dir = f'{args.outf}/model'
    os.makedirs(para_dir, exist_ok=True)
    modelpara_filename = os.path.join(para_dir, 'parameter.npz')
    np.savez(modelpara_filename,color_meanofdata=color_mean,geo_meanofdata=geo_mean,
                                color_model_mean=color_patch_mean, geo_model_mean=geo_patch_mean,
                                color_model_std=color_patch_std, geo_model_std=geo_patch_std)
    print('!!!done!!!')

def main():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--filedirs',  default='../../dataset/sjtu', type=str, help='your data filedir')
    parser.add_argument('--dataset',  default='sjtu', type=str, help='input dataset')
    parser.add_argument('--outf', default='results/sjtu', type=str,  help='output filedir for estimate model parameters')
    parser.add_argument('--K', default=80000, type=int,  help='kdtree partition K')
    parser.add_argument('--exp_id', default='test', type=str)

    args = parser.parse_args()
    args.outf = os.path.join(args.outf, args.exp_id)
    estimatemodelpara(args)


if __name__ == '__main__':
    main()