#coding=utf-8
"""
sklearn gdbt  
2017-07-27 张洛阳 杭州
"""
import argparse
from model import gdbt_model, extra_tree_model, catboost, xgboost_model
import datetime
import os
from data_utils import loadPredictData, convertPredictDataFormate, writeLabelProb



def parse_args():
    parser = argparse.ArgumentParser(description="DGBT用于识别虚假交易(自动搜索合适的参数 张洛阳 2017-07-27)")
    parser.add_argument("--date", help="训练或者预测时候的运行日期", default="2017-07-31", type=str)
    parser.add_argument("--n_estimators", help="随机森林当中树的数目 默认值 [100, 200, 300, 400, 500]", default=[100, 200, 300, 400, 500], type=list)
    parser.add_argument("--max_depth", help="树的最大深度 默认值[4, 5, 6, 7, 8, 9]", default=[4, 5, 6, 7, 8, 9], type=list)
    parser.add_argument("--min_samples_split", help="允许分裂的最小样本数目 默认值[20, 40, 60, 80]", default=[20, 40, 60, 80], type=list)
    parser.add_argument("--min_samples_leaf", help="叶子节点最小样本数目 默认值[10, 20, 30, 40, 50, 60, 70]", default=[40], type=list)
    parser.add_argument("--max_features", help="随机选取的最大特征数目(这个值要根据训练样本的特征数目选择) 默认值 'sqrt'", default=['sqrt'], type=list)
    parser.add_argument("--subsample", help="随机选择的样本数目(0~1之间) 默认值[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]", default=[0.6, 0.7, 0.8, 0.9], type=list)
    parser.add_argument("--random_state", help="随机数种子,  默认值100", default=100, type=int)
    parser.add_argument("--test_size", help="训练集和测试集比例划分, 默认值0.2", default=0.2, type=float)
    parser.add_argument("--n_jobs", help="参数选择的时启动的job数目,默认值4", default=8, type=int)
    parser.add_argument("--train", help="训练测试或者是预测, 默认值train (后选项 predict)", default="train", type=str)
    parser.add_argument("--train_file", help="训练集样本 libsvm数据格式", default="localtmp/h5_history_order_cheat/h5_uid_history_order_to_predict.data2017-07-27train", type=str)
    parser.add_argument("--model_file", help="训练生成的模型文件", default="gdbt.model", type=str)
    parser.add_argument("--encode_model_file", help="训练生成的编码模型文件", default="encode_gdbt.model", type=str)
    parser.add_argument("--params_file", help="训练生成的参数文件", default="gdbt_params.json", type=str)
    parser.add_argument("--predict_data", help="待预测的数据文件", default="localtmp/h5_history_order_cheat/h5_uid_history_order_to_predict.data2017-07-27", type=str)
    return parser.parse_args()


def train(date,
          train_file,
          random_state,
          test_size,
          n_estimators,
          max_depth,
          min_samples_split,
          min_samples_leaf,
          max_features,
          subsample,
          params_file,
          model_file,
          encode_model_file,
          n_jobs
          ):
    model = gdbt_model(train_file=train_file,
                       random_state=random_state,
                       test_size=test_size,
                       n_estimators=[100, 200, 300, 400, 500],
                       max_depth=[4, 5, 6, 7, 8, 9],
                       min_samples_split=[20, 40, 60, 80],
                       min_samples_leaf=[40],
                       max_features=["sqrt"],
                       subsample=[0.6, 0.7, 0.8, 0.9],
                       params_file="gdbt_params{}.json".format(date),
                       n_jobs=n_jobs)
    model.process()
    model.train(date=date)
    model.save(model_file="gbdt{}.model".format(date),
               encode_model_file="gdbt_encode{}.model".format(date))
    model.trainEvaluation()
    model2 = extra_tree_model(train_file=train_file,
                              random_state=random_state,
                              test_size=test_size,
                              min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf,
                              max_depth=[4, 5, 6, 7, 8, 9],
                              n_jobs=n_jobs,
                              params_file="extra_params{}.json".format(date)
                              )
    model2.process()
    model2.train(date=date)
    model2.save(model_file="extral{}.model".format(date),
               encode_model_file="extral{}.model".format(date))
    model2.trainEvaluation()
    model3 = catboost(
        train_file=train_file,
        params_file="catboost_params{}.json".format(date),
        random_state=random_state,
        test_size=test_size,
        iterations=[500],
        depth=[6],
        rsm = [0.8],
        n_jobs=n_jobs
    )
    model3.process()
    model3.train(date=date)
    model3.save("catboost{}.model".format(date))
    model4 = xgboost_model(train_file=train_file,
                           random_state=random_state,
                           test_size=test_size,
                           max_depth=[4, 5, 6, 7, 8, 9],
                           min_child_weight=[2, 4, 6, 8],
                           gamma = [0.0, 0.1, 0.2, 0.3, 0.4],
                           subsample=[0.6, 0.7, 0.8, 0.9],
                           colsample_bytree=[0.6, 0.7, 0.8, 0.9],
                           reg_alpha=[1e-5, 1e-2, 0.1],
                           n_estimators=[100, 200, 300, 400, 500],
                           params_file="xgboost_params{}.json".format(date),
                           n_jobs=n_jobs
                          )
    model4.process()
    model4.train(date=date)
    model4.save(model_file="xgboost{}.model".format(date),
                encode_model_file="xgboost_encode{}.model".format(date))
    model4.trainEvaluation()


def predict(train_file,
            random_state,
            test_size,
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            subsample,
            params_file,
            n_jobs,
            predict_data,
            model_file,
            encode_model_file
            ):
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
    model.load(model_file=model_file,
               encode_model_file=encode_model_file
               )
    convertPredictDataFormate(predict_data, "{}.converted".format(predict_data), "{}.id".format(predict_data))
    X, _ = loadPredictData("{}.converted".format(predict_data))
    predict_labels = model.predict(X)
    predict_probs = model.predict_prob(X)
    output_file = "{}.label.probe".format(predict_data)
    writeLabelProb(labels=predict_labels,
                   probs=predict_probs,
                   outputFile=output_file
                   )


def main(args):
    if not os.path.exists("log"):
        os.mkdir("log")
    if not os.path.exists("conf"):
        os.mkdir("conf")
    if not os.path.exists("model"):
        os.mkdir("model")
    if not os.path.exists("encode_model"):
        os.mkdir("encode_model")
    date = args.date
    n_estimators = args.n_estimators
    max_depth = args.max_depth
    min_samples_split = args.min_samples_split
    min_samples_leaf = args.min_samples_leaf
    max_features = args.max_features
    subsample = args.subsample
    random_state = args.random_state
    test_size = args.test_size
    n_jobs = args.n_jobs
    train_or_predict = args.train
    train_file = args.train_file
    model_file = args.model_file
    params_file = args.params_file
    encode_model_file = args.encode_model_file
    predict_data = args.predict_data
    if train_or_predict == "train":
        train(date=date,
              train_file=train_file,
              random_state=random_state,
              test_size=test_size,
              n_estimators=n_estimators,
              max_depth=max_depth,
              min_samples_split=min_samples_split,
              min_samples_leaf=min_samples_leaf,
              max_features=max_features,
              subsample=subsample,
              params_file=params_file,
              model_file=model_file,
              encode_model_file=encode_model_file,
              n_jobs=n_jobs
              )
    elif train_or_predict == "predict":
        predict(train_file=train_file,
                random_state=random_state,
                test_size=test_size,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                subsample=subsample,
                params_file=params_file,
                n_jobs=n_jobs,
                predict_data=predict_data,
                model_file=model_file,
                encode_model_file=encode_model_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)