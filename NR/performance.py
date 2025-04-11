import argparse, os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import csv
from tqdm import tqdm

def logistic_func(X, bayta1, bayta2, bayta3,bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X-bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat
def fit(test_data, mos):
    beta = [np.max(mos),np.min(mos),np.mean(test_data), np.std(test_data)]
    popt, _ = curve_fit(logistic_func, test_data, mos, p0=beta,maxfev=100000000)
    y_out = logistic_func(test_data, *popt)
    return y_out
def load_quality(test_data_file):
    test_data = np.load(test_data_file)
    test_result = test_data['quality']
    file_paths = test_data['file_paths']
    index = file_paths[0].find("_", 0) 
    name = file_paths[0][0:index]
    return test_result, name
def get_data(args, filename, name,p):
    if args.dataset == 'sjtu':
        score_data = pd.read_csv(filename)
        score = score_data[name]

        mos_data = pd.read_csv(args.mos_dir)
        mos = mos_data[name]

    elif args.dataset == 'ls-pcqa':
        if 'sel' in args.mos_dir:
            dex = [0,170,350,510,670,840]
        elif 'all' in args.mos_dir:
            dex = [0,185,375,560,745,930]
        score = []
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                score.append(float(row[0]))

        mos = []
        with open(args.mos_dir, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                mos.append(float(row[1]))
        mos = np.array(mos)
        mos = mos[dex[p]:dex[p+1]]

    elif args.dataset == 'icip' or  args.dataset == 'wpc':
        mos_data = pd.read_csv(args.mos_dir)
        total_obj_names = mos_data['name']
        mos_data = pd.read_csv(args.mos_dir,index_col = 0)

        obj_names = []
        mos = []
        for obj in total_obj_names: 
            if name in obj: 
                obj_names.append(obj)
        for i in obj_names:
            mos.append(mos_data.loc[i,:].tolist()[0])
        mos = np.array(mos)

        score_data = pd.read_csv(filename)
        score = score_data[name]
    score = fit(score, mos)
    return score, mos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sjtu', type=str, help='input dataset')
    parser.add_argument('--test_data_dir',  default='results',
                        type=str, help='test result filedir')
    args = parser.parse_args()
    args.mos_dir = os.path.join('../testset',args.dataset,'mos.csv')
    if args.dataset == 'ls-pcqa': args.mos_dir = os.path.join('../testset',args.dataset,'mos_all.csv')
    def get_score(test_dir,files,p):
        results = {}
        for file in files:
            test_data_file = os.path.join(test_dir, file,'result/result.npz')
            test_result, name = load_quality(test_data_file)
            results[name] = test_result
        dataframe = pd.DataFrame(results)

        csv_dir = f'{args.test_data_dir}/csv' 
        if not os.path.isdir(csv_dir):
           os.makedirs(csv_dir)
        filename=os.path.join(csv_dir, args.dataset+'.csv')
        dataframe.to_csv(filename,index=False,sep=',')

        with open(filename, "r") as file:
            reader = csv.reader(file)
            header = next(reader)
            test_pc = 0
            for i in header:
                score, single_mos = get_data(args,filename,i,p)
                if test_pc == 0:
                    test_score = np.array(score)
                    mos = np.array(single_mos)
                else:
                    test_score = np.concatenate((test_score,score), axis=0)
                    mos = np.concatenate((mos,single_mos), axis=0)
                test_pc = test_pc + 1
        return test_score,mos
    
    test_dir = os.path.join(args.test_data_dir, args.dataset)
    if args.dataset == 'wpc':
        common_names = [["banana", "cauliflower", "mushroom", "pineapple"],
                        ["bag", "biscuits", "cake", "flowerpot"],
                        ["glasses_case", "honeydew_melon", "house", "pumpkin"],
                        ["litchi", "pen_container", "ping-pong_bat", "puer_tea"],
                        ["ship", "statue", "stone", "tool_box"]]
        file_list = [[f"wpc_{name}_K80000_attri0.7" for name in group]for group in common_names]
    elif args.dataset == 'sjtu' or args.dataset == 'icip' or args.dataset == 'ls-pcqa':
        file_list = [[x] for x in sorted(os.listdir(test_dir))]
        
    PLCC, SROCC = [], []
    p = 0
    for files in tqdm(file_list):
        print(files)
        score, mos = get_score(test_dir,files,p)
        p+=1

        plcc, srcc = map(lambda func: func(score, mos)[0], [stats.pearsonr, stats.spearmanr])

        print(f"PLCC: {plcc:.4f}, SROCC: {srcc:.4f}")

        PLCC.append(plcc)
        SROCC.append(srcc)
        
    print(f"Average PLCC: {np.mean(PLCC):.4f}, Average SROCC: {np.mean(SROCC):.4f}")

if __name__ == '__main__':
    main()
