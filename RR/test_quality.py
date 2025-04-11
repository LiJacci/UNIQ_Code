import argparse, os, re
import numpy as np
from tqdm import tqdm
import pandas as pd
from feature_extract import get_feature_vector_data, compute_knn_density
from utils import split_pc,load_point_cloud,update_csv_func

def score(x,y):
    s = np.abs(x-y)
    return s

### compute quality
def distance(args, para, test, delet_j,dens_one):
    distance = []
    for i in range(args.num_cube):
        if i not in delet_j:
            para_features, test_features = para[:,i], test[:,i]
            x1, x2 = para_features[:int(args.feature_num/2)], para_features[int(args.feature_num/2):]
            y1, y2 = test_features[:int(args.feature_num/2)], test_features[int(args.feature_num/2):]
            attri = np.mean(score(x1,y1))
            geo = np.mean(score(x2,y2))
            quality = args.attri_weight*attri + (1 - args.attri_weight)*geo
            quality = quality/dens_one[i]
            distance.append(quality)
    if delet_j is not None: print(len(distance)==(args.num_cube-len(delet_j)))
    distance = np.mean(distance)
    return distance

def computequality(args, coord, feats, filename,filedir):

    ## divide specific number of patches
    para = np.load(args.modelpara_path)
    box = np.array(para['box'])
    coord_near, feats_near = split_pc(coord,feats,box,args)
    S = len(coord_near)

### compute features of each patch        
    print('Compute features for \n', filename)
    from joblib import Parallel, delayed
    delet_j = []
    def process_patch(j,delet_j):
        coord_j = coord_near[j].reshape(-1,3)
        feats_j = feats_near[j].reshape(-1,3)
        if coord_j.shape[0] <= 1:
            return np.zeros((int(args.feature_num/2), 1)), np.zeros((int(args.feature_num/2), 1)), [j], np.array([0])
        else:
            color_params,geo_params = get_feature_vector_data(coord_j,feats_j)
            color = np.expand_dims(color_params,1)
            geo = np.expand_dims(geo_params,1)
            den_j = compute_knn_density(coord_j)
            return color, geo, [],np.array([den_j])

    results = Parallel(n_jobs=-1)(delayed(process_patch)(j,delet_j) for j in tqdm(range(S)))
    color_one = np.concatenate([res[0] for res in results], axis=1) #(feature_num, patches)
    geo_one = np.concatenate([res[1] for res in results], axis=1)
    delet_j = [j for res in results for j in res[2]]
    dens_one = np.concatenate([res[3] for res in results], axis=0)

### load pristine paras
    para_features = np.array(para['features'],dtype=np.float16)
    feature = np.concatenate((color_one, geo_one), axis=0)
    quality = distance(args, para_features, feature, delet_j,dens_one)
    quality = quality
    return quality

def main():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--filedirs',  default='../../testset/ls-pcqa/pc', type=str, help='input distortion point cloud filedir')
    parser.add_argument('--dataset',  default='ls-pcqa', type=str, help='input dataset')
    parser.add_argument('--modelpara', default='../../RR/results/0408_paramode_1/ls-pcqa/ls-pcqa_*_featurenum40_patch512/model/para.npz', type=str,  help='filedir for saving the model para')
    parser.add_argument('--outf', default='results/ls-pcqa', type=str,  help='output filedir for save patches of test pc')
    parser.add_argument('--num_cube', default=512, type=int,  help='cube numbers')
    parser.add_argument('--feature_num', default=40, type=int,  help='the numbers of features')
    parser.add_argument('--attri_weight', default=0.7,type=float, help='the weight for attribute')
    args = parser.parse_args()

### load test distortion point cloud

    test_list = os.listdir(args.filedirs)
    test_pc = [item for item in test_list if os.path.isdir(os.path.join(args.filedirs, item))]

    def process_pc(args, pc, file_paths, update_func=None):
        quality_one = []
        print('point cloud:', pc)
        args.modelpara_path = args.modelpara.replace('*', pc)
        exp_id = f'{args.dataset}_{pc}_patch{args.num_cube}'
        args.outfile = os.path.join(args.outf, exp_id)
        result_dir = os.path.join(args.outfile, 'result')

        if os.path.isdir(result_dir):
            return

        for file in tqdm(file_paths):
            filename = file.split('.')[0]
            filedir = os.path.join(args.filedirs, pc, file)
            coord, feats = load_point_cloud(filedir)

            print('Compute features for \n', filename)
            quality = computequality(args, coord, feats, filename, filedir)
            quality_one.append(quality)

            if update_func:
                update_func(filename, quality)

        os.makedirs(result_dir, exist_ok=True)
        np.savez(os.path.join(result_dir, 'result.npz'), quality=quality_one, file_paths=file_paths)

    for pc in tqdm(test_pc):
        pc_dir = os.path.join(args.filedirs, pc)
        file_paths = sorted(os.listdir(pc_dir))

        if args.dataset == 'sjtu':
            file_paths.sort(key=lambda x: int(re.search(r'\d+', x).group()))
            process_pc(args, pc, file_paths)

        elif args.dataset in ['icip', 'wpc']:
            process_pc(args, pc, file_paths)

        elif args.dataset == 'ls-pcqa':
            csv_path = os.path.join(args.outf, 'score.csv')
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            if not os.path.exists(csv_path):
                pd.DataFrame(columns=["filename", "quality"]).to_csv(csv_path, index=False)
            update_func = update_csv_func(csv_path)
            process_pc(args, pc, file_paths, update_func)
        #####for ls-pcqa, you should divide these 930 pc into 85 folders according to reference names

if __name__ == '__main__':
    main()
