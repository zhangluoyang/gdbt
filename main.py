#coding=utf-8
"""
sklearn gdbt  
2017-07-27 张洛阳 杭州
"""
import argparse
import json
from model import gdbt_model
import os
def parse_args():
    parser = argparse.ArgumentParser(description="DGBT用于识别虚假交易(自动搜索合适的参数 张洛阳 2017-07-27)")
    parser.add_argument("--n_estimators", help="随机森林当中树的数目 默认值 [100, 200, 300, 400, 500]", default=[100, 200, 300, 400, 500], type=list)
    parser.add_argument("--max_depth", help="树的最大深度 默认值[4, 5, 6, 7, 8, 9]", default=[4, 5, 6, 7, 8, 9], type=list)
    parser.add_argument("--min_samples_split", help="允许分裂的最小样本数目 默认值[20, 40, 60, 80]", default=[20, 40, 60, 80], type=list)
    parser.add_argument("--min_samples_leaf", help="叶子节点最小样本数目 默认值[10, 20, 30, 40, 50, 60, 70]", default=[10, 20, 30, 40, 50], type=list)
    parser.add_argument("--max_features", help="随机选取的最大特征数目(这个值要根据训练样本的特征数目选择) 默认值 'sqrt'", default=['sqrt'], type=list)
    parser.add_argument("--subsample", help="随机选择的样本数目(0~1之间) 默认值[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]", default=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], type=list)
    parser.add_argument("--random_state", help="随机数种子,  默认值100", default=100, type=int)
    parser.add_argument("--test_size", help="训练集和测试集比例划分, 默认值0.2", default=0.2, type=float)
    parser.add_argument("--n_jobs", help="参数选择的时启动的job数目,默认值4", default=8, type=int)
    parser.add_argument("--train", help="训练测试或者是预测, 默认值train (后选项 eval,predict)", default="train", type=str)
    parser.add_argument("--train_file", help="训练集样本 libsvm数据格式", default="datas.txt", type=str)
    parser.add_argument("--model_file", help="训练生成的模型文件", default="model/gdbt.model", type=str)
    parser.add_argument("--params_file", help="训练生成的参数文件", default="model/gdbt_params.json", type=str)
    return parser.parse_args()

def main(args):
    if not os.path.exists("log"):
        os.mkdir("log")
    if not os.path.exists("conf"):
        os.mkdir("conf")
    if not os.path.exists("model"):
        os.mkdir("model")
    n_estimators = args.n_estimators
    max_depth = args.max_depth
    min_samples_split = args.min_samples_split
    min_samples_leaf = args.min_samples_leaf
    max_features = args.max_features
    subsample = args.subsample
    random_state = args.random_state
    test_size = args.test_size
    n_jobs = args.n_jobs
    train = args.train
    train_file = args.train_file
    model_file = args.model_file
    params_file = args.params_file
    if train == "train":
        model = gdbt_model(train_file = train_file,
                   random_state = random_state,
                   test_size = test_size,
                   n_estimators = n_estimators,
                   max_depth = max_depth,
                   min_samples_split = min_samples_split,
                   min_samples_leaf = min_samples_leaf,
                   max_features = max_features,
                   subsample = subsample,
                   params_file = params_file,
                   n_jobs = n_jobs)
        model.process()
        model.train()
        model.save(model_file=model_file)
    elif train == "predict":
        model = gdbt_model(train_file=train_file,
                           random_state=random_state,
                           test_size=test_size,
                           n_estimators=n_estimators,
                           max_depth=max_depth,
                           min_samples_split=min_samples_split,
                           min_samples_leaf=min_samples_leaf,
                           max_features=max_features,
                           subsample=subsample,
                           params_file=params_file,
                           n_jobs=n_jobs)
        model.load(model_file=model_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)