import argparse, os, glob
import numpy as np
from pyntcloud import PyntCloud
from tqdm import tqdm
from feature_extract import get_feature_vector_data
from utils import kdtree_partition, split_pc

def load_data(args, filedir):
        cloud = PyntCloud.from_file(filedir)
        coord = np.array(cloud.points)[:,0:3]
        if args.dataset == 'sjtu':
           feats = np.array(cloud.points)[:,6:9]
        else: feats = np.array(cloud.points)[:,3:6]
        return coord, feats

def estimatemodelpara(args):   
### load dataset
    filedir = args.filename
    filename = os.path.splitext(os.path.basename(filedir))[0]
    print('load point cloud:', filename)
    coord, feats = load_data(args, filedir)
    ##### 1) partition by KD-Tree
    point_num = coord.shape[0]
    num_points = int(point_num/args.num_cube)+2
    coord_near, _ = kdtree_partition(points=coord,feats=feats,num_points=num_points)
    S = coord_near.shape[0]
    ##### 2) get bounding box
    box = []
    for j in tqdm(range(S)):
        coord_near_j = coord_near[j,:,:].reshape(-1,3)
        x1,x2 = np.min(coord_near_j[:,0]), np.max(coord_near_j[:,0])
        y1,y2 = np.min(coord_near_j[:,1]), np.max(coord_near_j[:,1])
        z1,z2 = np.min(coord_near_j[:,2]), np.max(coord_near_j[:,2])
        box.append(np.array([x1, x2, y1, y2, z1, z2]))

    ##### 3) partition by bounding box
    coord_near, feats_near = split_pc(coord,feats,box,args)
    S = len(coord_near)

    ##### 4) compute features for each block
    print('Compute features for:', filename)
    from joblib import Parallel, delayed
    def process_patch(j):
        coord_near_j = coord_near[j].reshape(-1,3)
        feats_near_j = feats_near[j].reshape(-1,3)
        if coord_near_j.shape[0] <= 1:
            print('empty!!!',j)
            return np.zeros((int(args.feature_num/2), 1)), np.zeros((int(args.feature_num/2), 1))
        else:
            color_params, geo_params = get_feature_vector_data(coord_near_j,feats_near_j)
            color = np.expand_dims(color_params,1) 
            geo = np.expand_dims(geo_params,1)
            return color,geo
    results = Parallel(n_jobs=-1)(delayed(process_patch)(j) for j in tqdm(range(S)))
    color_one = np.concatenate([res[0] for res in results], axis=1) #(feature_num, patches)
    geo_one = np.concatenate([res[1] for res in results], axis=1)

    ##### 5) save modelpara and bounding box
    para_dir = f'{args.outfile}/model'
    if not os.path.isdir(para_dir):
        os.makedirs(para_dir)
    modelpara_filename = os.path.join(para_dir,'para.npz')
    feature = np.concatenate((color_one, geo_one), axis=0)
    print(feature.shape)
    np.savez(modelpara_filename, features = feature, box=box)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filedir',  default='../../dataset/sjtu/*.ply', type=str, help='input filedir for estimate model parameters')
    parser.add_argument('--dataset',  default='sjtu', type=str, help='input dataset')
    parser.add_argument('--outf', default='results/sjtu', type=str,  help='output filedir for estimate model parameters') #try2
    parser.add_argument('--num_cube', default=512, type=int,  help='cube numbers')
    parser.add_argument('--feature_num', default=40, type=int,  help='the numbers of features')
    args = parser.parse_args()
    
    ply_files = glob.glob(args.filedir)
    ply_files = ply_files[0:2]
    for ply in tqdm(ply_files):
        args.filename = os.path.abspath(ply)
        testpc = os.path.basename(ply).replace('.ply', '')
        exp_id = f'{args.dataset}_{testpc}_patch{args.num_cube}'
        args.outfile = os.path.join(args.outf, exp_id)
        print('outfile:',args.outfile)
        if os.path.isdir(args.outfile):
            continue
        
        estimatemodelpara(args)


if __name__ == '__main__':
    main()