import argparse
import os
import numpy as np
from tqdm import tqdm
from pyntcloud import PyntCloud
from feature_extract import get_feature_vector_data,compute_knn_density
from utils import sort_points, kdtree_partition


### compute Mahalanobis distance
def d_equ(para_mean, para_std, test_mean, test_std):
    invcov_param = np.linalg.pinv((test_std + para_std) / 2)
    d = np.matmul(np.matmul((para_mean - test_mean), invcov_param), np.transpose((para_mean - test_mean)))
    d = float(np.squeeze(np.sqrt(d)))

    return d

def computequality(args, coord, feats, filename):
    coord, feats = sort_points(coord, feats, n_bits=10)

    ## block partition
    coord_block, feats_block = kdtree_partition(coord, feats, num_points=args.K)
    S = coord_block.shape[0]
    if S <= 2:
        coord_block, feats_block = kdtree_partition(coord, feats, num_points=int((coord.shape[0]) / 3))
    coord_block = [coord_block[p, :, :] for p in range(coord_block.shape[0])]
    feats_block = [feats_block[q, :, :] for q in range(feats_block.shape[0])]
    print('Number of patches:', len(coord_block))

    ### compute features of each patch
    print('Compute features for \n', filename)
    from joblib import Parallel, delayed
    def process_patch(j):
        coord_j = coord_block[j].reshape(-1, 3)
        feats_j = feats_block[j].reshape(-1, 3)
        dens = compute_knn_density(coord_j)
        color_params, geo_params = get_feature_vector_data(coord_j, feats_j)
        color = np.expand_dims(color_params, 1)
        geo = np.expand_dims(geo_params, 1)
        return color, geo, dens

    results = Parallel(n_jobs=-1)(delayed(process_patch)(j) for j in range(len(coord_block)))
    color_one = np.concatenate([res[0] for res in results], axis=1)  # (feature_num, patches)
    geo_one = np.concatenate([res[1] for res in results], axis=1)
    dens_one = [res[2] for res in results]
    
    ### compute quality
    # mean of test color and all geo
    para = np.load(args.modelpara)

    def centerized(params, meanofdata):
        meanMatric = np.tile(meanofdata, (1, params.shape[1]))
        centerlized = params - meanMatric
        return centerlized

    color_one = centerized(color_one, para['color_meanofdata'])
    geo_one = centerized(geo_one, para['geo_meanofdata'])
    dens_test_mean = sum(dens_one) / len(dens_one)

    def cross_block(color_one,geo_one):
        color_test_mean = np.nanmean(color_one, axis=1) #(feature_num,)
        geo_test_mean = np.nanmean(geo_one, axis=1) #(feature_num,)
        distparam_no_nan_color = color_one[:,~np.isnan(color_one).any(axis=0)]
        color_test_std = np.cov(distparam_no_nan_color, rowvar=True)
        distparam_no_nan_geo = geo_one[:,~np.isnan(geo_one).any(axis=0)]
        geo_test_std = np.cov(distparam_no_nan_geo, rowvar=True)
        return color_test_mean,geo_test_mean,color_test_std,geo_test_std
    color_test_mean,geo_test_mean,color_test_std,geo_test_std = cross_block(color_one,geo_one)
    col = d_equ(para['color_model_mean'],para['color_model_std'],color_test_mean,color_test_std)
    geo = d_equ(para['geo_model_mean'],para['geo_model_std'],geo_test_mean,geo_test_std)

    d = args.attri_weight*col + (1 - args.attri_weight)*geo
    quality = d / dens_test_mean

    return quality

def main():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--dataset', default='sjtu', type=str, help='input dataset')
    parser.add_argument('--datafile', default='../../testset/sjtu/longdress', type=str, help='input dataset filedir')
    parser.add_argument('--testname', default='longdress', type=str, help='input data name')
    parser.add_argument('--modelpara',default='../../20250331_ljx/results/0409/RWTT/model/parameter.npz',
                        type=str, help='filedir for saving the model para')
    parser.add_argument('--outf', default='results/sjtu', type=str, help='output filedir')
    parser.add_argument('--K', default=80000, type=int, help='kdtree partition K')
    parser.add_argument('--attri_weight', default=0.7, type=float, help='the weight for attribute')

    args = parser.parse_args()

    ### load test distortion point cloud
    def test_dir(args,outf, filedirs, file_paths):
        quality_one = []
        file_paths = file_paths[0:5]
        for file in tqdm(file_paths):
            index = file.find(".", 0)  # find the position of '.'
            filename = file[0:index]  # get the filename
            print('load point cloud', filename)
            filedir = os.path.join(filedirs, file)
            cloud = PyntCloud.from_file(filedir)
            coord = np.array(cloud.points)[:,0:3]
            feats = np.array(cloud.points)[:,3:6]
            print('Compute features for \n', filename)
            quality = computequality(args, coord, feats, filename) #
            quality_one.append(quality)
            
        result_dir = f'{outf}/result'
        if not os.path.isdir(result_dir):
           os.makedirs(result_dir)
        result_filename = os.path.join(result_dir,'result.npz')
        np.savez(result_filename, quality = quality_one, file_paths = file_paths)
    
    if args.dataset == 'sjtu':
        file_paths = os.listdir(args.datafile)
        import re
        file_paths.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        print('dbg:', len(file_paths))
        exp_id = f'{args.dataset}_{args.testname}_K{args.K}_attri{args.attri_weight}'
        outf = os.path.join(args.outf, exp_id)
        test_dir(args,outf,args.datafile,file_paths)
    else:
        file_paths = os.listdir(args.datafile)
        file_paths = sorted(file_paths)
        print('dbg:', len(file_paths))
        exp_id = f'{args.dataset}_{args.testname}_K{args.K}_attri{args.attri_weight}'
        outf = os.path.join(args.outf, exp_id)
        test_dir(args,outf,args.datafile,file_paths)


if __name__ == '__main__':
    main()