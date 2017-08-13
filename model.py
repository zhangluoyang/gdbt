#coding=utf-8
"""
sklearn gdbt
2017-07-30 张洛阳 杭州
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import logging
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from xgboost.sklearn import XGBClassifier
import json


class gdbt_model(object):

    def __init__(self,
                 train_file,
                 random_state,
                 test_size,
                 n_estimators,
                 max_depth,
                 min_samples_split,
                 max_features,
                 min_samples_leaf,
                 subsample,
                 params_file,
                 n_jobs):
        self.train_file = train_file
        self.random_state = random_state
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.params_file = params_file
        self.n_jobs = n_jobs


    def process(self):
        X, y = load_svmlight_file(self.train_file)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            random_state=self.random_state,
                                                            test_size=self.test_size)
        self.X_train = X_train.toarray()
        self.y_train = y_train

        self.X_test = X_test.toarray()
        self.y_test = y_test


    def train(self,
              date):
        logging.basicConfig(filename="log/{}gdbtTrain.log".format(date), format="%(message)s",
                            filemode="a",
                            level=logging.INFO)
        param_test = {'n_estimators': self.n_estimators, 'learning_rate': [0.005, 0.01]}
        gsearch = GridSearchCV(
            estimator=GradientBoostingClassifier(min_samples_split=500,
                                                 min_samples_leaf=50,
                                                 max_depth=8,
                                                 max_features='sqrt',
                                                 subsample=0.8,
                                                 random_state=self.random_state),
            param_grid=param_test, scoring='roc_auc', n_jobs=self.n_jobs, iid=False, cv=5)
        gsearch.fit(self.X_train, self.y_train)
        logging.info("search n_estimators result ......................")
        logging.info(gsearch.grid_scores_)
        logging.info(gsearch.best_params_)
        logging.info("score:{}".format(gsearch.best_score_))
        best_n_estimators = gsearch.best_params_['n_estimators']
        best_learning_rate = gsearch.best_params_['learning_rate']

        param_test = {'max_depth': self.max_depth, 'min_samples_split': self.min_samples_split}
        gsearch = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=best_learning_rate,
                                                 n_estimators=best_n_estimators,
                                                 max_features='sqrt',
                                                 subsample=0.8,
                                                 random_state=self.random_state),
            param_grid=param_test, scoring='roc_auc', n_jobs=self.n_jobs, iid=False, cv=5)
        gsearch.fit(self.X_train, self.y_train)
        logging.info("search max_depth and min_samples_split result ......................")
        logging.info(gsearch.grid_scores_)
        logging.info(gsearch.best_params_)
        logging.info("score:{}".format(gsearch.best_score_))
        best_max_depth = gsearch.best_params_['max_depth']
        best_min_samples_split = gsearch.best_params_['min_samples_split']

        param_test = {'min_samples_leaf': self.min_samples_leaf}
        gsearch = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=best_learning_rate,
                                                 n_estimators=best_n_estimators,
                                                 subsample=0.8,
                                                 random_state=self.random_state,
                                                 max_depth=best_max_depth,
                                                 min_samples_split=best_min_samples_split,
                                                 ),
            param_grid=param_test, scoring='roc_auc', n_jobs=self.n_jobs, iid=False, cv=5)
        gsearch.fit(self.X_train, self.y_train)
        logging.info("search min_samples_split result ......................")
        logging.info(gsearch.grid_scores_)
        logging.info(gsearch.best_params_)
        logging.info("score:{}".format(gsearch.best_score_))
        best_min_samples_leaf = gsearch.best_params_['min_samples_leaf']

        param_test = {'max_features': self.max_features}
        gsearch = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=best_learning_rate,
                                                 n_estimators=best_n_estimators,
                                                 subsample=0.8,
                                                 random_state=self.random_state,
                                                 max_depth=best_max_depth,
                                                 min_samples_split=best_min_samples_split,
                                                 min_samples_leaf=best_min_samples_leaf),
            param_grid=param_test, scoring='roc_auc', n_jobs=self.n_jobs, iid=False, cv=5)
        gsearch.fit(self.X_train, self.y_train)
        logging.info("search max_features result ......................")
        logging.info(gsearch.grid_scores_)
        logging.info(gsearch.best_params_)
        logging.info("score:{}".format(gsearch.best_score_))
        best_max_features = gsearch.best_params_['max_features']

        param_test = {'subsample': self.subsample}
        gsearch = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=best_learning_rate,
                                                 n_estimators=best_n_estimators,
                                                 max_features=best_max_features,
                                                 random_state=self.random_state,
                                                 max_depth=best_max_depth,
                                                 min_samples_split=best_min_samples_split),
            param_grid=param_test, scoring='roc_auc', n_jobs=self.n_jobs, iid=False, cv=5)
        gsearch.fit(self.X_train, self.y_train)
        logging.info("search subsample result ......................")
        logging.info(gsearch.grid_scores_)
        logging.info(gsearch.best_params_)
        logging.info("score:{}".format(gsearch.best_score_))
        best_subsample = gsearch.best_params_['subsample']

        best_params = {"best_learning_rate":best_learning_rate,
                       "best_n_estimators":best_n_estimators,
                       "best_max_features":best_max_features,
                       "best_min_samples_leaf":best_min_samples_leaf,
                       "best_max_depth":best_max_depth,
                       "best_min_samples_split":best_min_samples_split,
                       "best_subsample":best_subsample,
                       "n_jobs":self.n_jobs
                       }
        json.dump(best_params, open("conf/{}".format(self.params_file), "w"))

        logging.info({"best_learning_rate".format(best_learning_rate)})
        logging.info("best_n_estimators:{}".format(best_n_estimators))
        logging.info("best_max_features:{}".format(best_max_features))
        logging.info("best_min_samples_leaf:{}".format(best_min_samples_leaf))
        logging.info("best_max_depth:{}".format(best_max_depth))
        logging.info("best_min_samples_split:{}".format(best_min_samples_split))
        logging.info("best_subsample:{}".format(best_subsample))

        self.best_estimator = gsearch.best_estimator_

        temp_model = gsearch.best_estimator_
        test_predprob = temp_model.predict_proba(self.X_test)[:, 1]
        logging.info("AUC Score (Test): %f" % metrics.roc_auc_score(self.y_test, test_predprob))

        # encode one-hot features
        grd_enc = OneHotEncoder()
        grd_enc.fit(temp_model.apply(self.X_train)[:, :, 0])
        self.grd_enc = grd_enc


    def save(self,
             model_file,
             encode_model_file
             ):
        joblib.dump(self.best_estimator, "model/{}".format(model_file))
        joblib.dump(self.grd_enc, "encode_model/{}".format(encode_model_file))

    def load(self,
             model_file,
             encode_model_file
             ):
        self.best_estimator = joblib.load("model/{}".format(model_file))
        self.grd_enc = joblib.load("encode_model/{}".format(encode_model_file))


    def predict_prob(self,
                     X):
        return self.best_estimator.predict_proba(X)


    def predict(self,
                X):
        return self.best_estimator.predict(X)


    def encode(self,
               X):
        return self.grd_enc.transform(X)


    def trainEvaluation(self):
        labels = self.y_test
        predicts = self.predict(self.X_test)
        self.evaluation(labels=labels,
                        predicts=predicts
                        )


    def evaluation(self, labels, predicts):
        assert (labels.shape == predicts.shape)
        m = labels.shape
        c11 = 0.0
        c10 = 0.0
        c01 = 0.0
        c00 = 0.0
        count0 = 0.0
        count1 = 0.0
        for idx in range(0, m[0]):
            if labels[idx] == 0:
                count0 += 1
            else:
                count1 += 1
            if labels[idx] == 0 and predicts[idx] == 0:
                c00 += 1
            elif labels[idx] == 0 and predicts[idx] == 1:
                c01 += 1
            elif labels[idx] == 1 and predicts[idx] == 0:
                c10 += 1
            else:
                c11 += 1
        logging.info("score: {}".format((c00 + c11) / float(c00 + c01 + c11 + c10 + 0.001)))
        logging.info("precision: {}".format(c11 / float(c11 + c01 + 0.001)))
        logging.info("recall: {}".format(c11 / float(c11 + c10 + 0.001)))
        logging.info("positive: {}, negative: {}".format(c01 + c11, c10 + c00))
        return c11, c10, c01, c00


class extra_tree_model(object):

    def __init__(self,
                 train_file,
                 random_state,
                 test_size,
                 min_samples_split,
                 min_samples_leaf,
                 max_depth,
                 n_jobs,
                 params_file,
                 max_features="auto",
                 min_weight_fraction_leaf=0,
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7
                 ):
        self.train_file = train_file
        self.random_state = random_state
        self.test_size = test_size
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.params_file = params_file
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_weight_fraction_leaf = max_depth
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_split = min_impurity_split

    def process(self):
        X, y = load_svmlight_file(self.train_file)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            random_state=self.random_state,
                                                            test_size=self.test_size)
        self.X_train = X_train.toarray()
        self.y_train = y_train

        self.X_test = X_test.toarray()
        self.y_test = y_test


    def train(self,
            date):
        logging.basicConfig(filename="log/{}extraTreeTrain.log".format(date), format="%(message)s",
                                filemode="a",
                                level=logging.INFO)

        param_test = {'max_depth': self.max_depth, 'min_samples_split': self.min_samples_split}
        gsearch = GridSearchCV(
            estimator=ExtraTreeClassifier(min_samples_leaf=50,
                                          max_features='auto',
                                          random_state=self.random_state),
                param_grid=param_test, scoring='roc_auc', n_jobs=self.n_jobs, iid=False, cv=5)
        gsearch.fit(self.X_train, self.y_train)
        logging.info("search max_depth and min_samples_split result ......................")
        logging.info(gsearch.grid_scores_)
        logging.info(gsearch.best_params_)
        logging.info("score:{}".format(gsearch.best_score_))
        best_max_depth = gsearch.best_params_['max_depth']
        best_min_samples_split = gsearch.best_params_['min_samples_split']

        param_test = {'min_samples_leaf': self.min_samples_leaf}
        gsearch = GridSearchCV(
            estimator=ExtraTreeClassifier(max_features='auto',
                                          max_depth = best_max_depth,
                                          min_samples_split=best_min_samples_split,
                                          random_state=self.random_state),
            param_grid=param_test, scoring='roc_auc', n_jobs=self.n_jobs, iid=False, cv=5)
        gsearch.fit(self.X_train, self.y_train)
        logging.info("search min_samples_leaf ......................")
        logging.info(gsearch.grid_scores_)
        logging.info(gsearch.best_params_)
        logging.info("score:{}".format(gsearch.best_score_))
        best_min_samples_leaf = gsearch.best_params_["min_samples_leaf"]

        best_params = {"best_max_depth": best_max_depth,
                       "best_min_samples_split": best_min_samples_split,
                       "best_min_samples_leaf": best_min_samples_leaf,
                       "n_jobs": self.n_jobs
                        }
        json.dump(best_params, open("conf/{}".format(self.params_file), "w"))

        self.best_estimator = gsearch.best_estimator_

        temp_model = gsearch.best_estimator_
        test_predprob = temp_model.predict_proba(self.X_test)[:, 1]
        logging.info("AUC Score (Test): %f" % metrics.roc_auc_score(self.y_test, test_predprob))

        # encode one-hot features
        grd_enc = OneHotEncoder()
        grd_enc.fit(temp_model.apply(self.X_train))
        self.grd_enc = grd_enc


    def save(self,
             model_file,
             encode_model_file
            ):
        joblib.dump(self.best_estimator, "model/{}".format(model_file))
        joblib.dump(self.grd_enc, "encode_model/{}".format(encode_model_file))


    def load(self,
             model_file,
             encode_model_file
            ):
        self.best_estimator = joblib.load("model/{}".format(model_file))
        self.grd_enc = joblib.load("encode_model/{}".format(encode_model_file))


    def predict_prob(self,
                     X):
        return self.best_estimator.predict_proba(X)


    def predict(self,
                X):
        return self.best_estimator.predict(X)


    def encode(self,
                X):
        return self.grd_enc.transform(X)


    def trainEvaluation(self):
        labels = self.y_test
        predicts = self.predict(self.X_test)
        self.evaluation(labels=labels,
                        predicts=predicts
                        )


    def evaluation(self, labels, predicts):
        assert (labels.shape == predicts.shape)
        m = labels.shape
        c11 = 0.0
        c10 = 0.0
        c01 = 0.0
        c00 = 0.0
        count0 = 0.0
        count1 = 0.0
        for idx in range(0, m[0]):
            if labels[idx] == 0:
                count0 += 1
            else:
                count1 += 1
            if labels[idx] == 0 and predicts[idx] == 0:
                c00 += 1
            elif labels[idx] == 0 and predicts[idx] == 1:
                c01 += 1
            elif labels[idx] == 1 and predicts[idx] == 0:
                c10 += 1
            else:
                c11 += 1
        logging.info("score: {}".format((c00 + c11) / float(c00 + c01 + c11 + c10 + 0.001)))
        logging.info("precision: {}".format(c11 / float(c11 + c01 + 0.001)))
        logging.info("recall: {}".format(c11 / float(c11 + c10 + 0.001)))
        logging.info("positive: {}, negative: {}".format(c01 + c11, c10 + c00))
        return c11, c10, c01, c00


class xgboost_model(object):

    def __init__(self,
                 train_file,
                 random_state,
                 test_size,
                 max_depth,
                 min_child_weight,
                 gamma,
                 subsample,
                 colsample_bytree,
                 reg_alpha,
                 n_estimators,
                 params_file,
                 n_jobs):
        self.train_file = train_file
        self.random_state = random_state
        self.test_size = test_size
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.n_estimators = n_estimators
        self.params_file = params_file
        self.n_jobs = n_jobs

    def process(self):
        X, y = load_svmlight_file(self.train_file)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            random_state=self.random_state,
                                                            test_size=self.test_size)
        self.X_train = X_train.toarray()
        self.y_train = y_train

        self.X_test = X_test.toarray()
        self.y_test = y_test

    def train(self,
              date):
        logging.basicConfig(filename="log/{}gxboostTrain.log".format(date), format="%(message)s",
                            filemode="a",
                            level=logging.INFO)
        param_test = {'max_depth':self.max_depth,
                      'min_child_weight':self.min_child_weight}
        gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                                        n_estimators=200,
                                                        max_depth=5,
                                                        min_child_weight=1,
                                                        gamma=0,
                                                        subsample=0.8,
                                                        colsample_bytree=0.8,
                                                        objective='binary:logistic',
                                                        nthread=self.n_jobs,
                                                        scale_pos_weight=1,
                                                        seed=self.random_state),
                                param_grid=param_test, scoring='roc_auc', iid=False, cv=5)

        gsearch.fit(self.X_train, self.y_train)
        logging.info("search max_depth and min_child_weight......................")
        logging.info(gsearch.grid_scores_)
        logging.info(gsearch.best_params_)
        logging.info("score:{}".format(gsearch.best_score_))
        best_max_depth = gsearch.best_params_['max_depth']
        best_min_child_weight = gsearch.best_params_['min_child_weight']

        param_test = {'gamma': self.gamma}
        gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                                        n_estimators=200,
                                                        max_depth=best_max_depth,
                                                        min_child_weight=best_min_child_weight,
                                                        subsample=0.8,
                                                        colsample_bytree=0.8,
                                                        objective='binary:logistic',
                                                        nthread=self.n_jobs,
                                                        scale_pos_weight=1,
                                                        seed=self.random_state),
                                param_grid=param_test, scoring='roc_auc', iid=False, cv=5)
        gsearch.fit(self.X_train, self.y_train)
        logging.info("search gamma......................")
        logging.info(gsearch.grid_scores_)
        logging.info(gsearch.best_params_)
        logging.info("score:{}".format(gsearch.best_score_))
        best_gamma = gsearch.best_params_['gamma']

        param_test = {'subsample': self.subsample,
                       'colsample_bytree': self.colsample_bytree}
        gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                                       n_estimators=200,
                                                       max_depth=best_max_depth,
                                                       min_child_weight=best_min_child_weight,
                                                       gamma=best_gamma,
                                                       objective='binary:logistic',
                                                       nthread=self.n_jobs,
                                                       scale_pos_weight=1,
                                                       seed=self.random_state),
                                param_grid=param_test, scoring='roc_auc', iid=False, cv=5)
        gsearch.fit(self.X_train, self.y_train)
        logging.info("search subsample and colsample_bytree......................")
        logging.info(gsearch.grid_scores_)
        logging.info(gsearch.best_params_)
        logging.info("score:{}".format(gsearch.best_score_))
        best_subsample = gsearch.best_params_['subsample']
        best_colsample_bytree = gsearch.best_params_['colsample_bytree']

        param_test = {'reg_alpha': self.reg_alpha}
        gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                                       n_estimators=200,
                                                       subsample=best_subsample,
                                                       colsample_bytree=best_colsample_bytree,
                                                       max_depth=best_max_depth,
                                                       min_child_weight=best_min_child_weight,
                                                       gamma=best_gamma,
                                                       objective='binary:logistic',
                                                       nthread=self.n_jobs,
                                                       scale_pos_weight=1,
                                                       seed=self.random_state),
                               param_grid=param_test, scoring='roc_auc', iid=False, cv=5)
        gsearch.fit(self.X_train, self.y_train)
        logging.info("search reg_alpha......................")
        logging.info(gsearch.grid_scores_)
        logging.info(gsearch.best_params_)
        logging.info("score:{}".format(gsearch.best_score_))
        best_reg_alpha = gsearch.best_params_['reg_alpha']

        param_test = {'learning_rate': [0.001, 0.005], "n_estimators":self.n_estimators}
        gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                                       n_estimators=200,
                                                       subsample=best_subsample,
                                                       colsample_bytree=best_colsample_bytree,
                                                       max_depth=best_max_depth,
                                                       min_child_weight=best_min_child_weight,
                                                       gamma=best_gamma,
                                                       objective='binary:logistic',
                                                       nthread=self.n_jobs,
                                                       scale_pos_weight=1,
                                                       seed=self.random_state,
                                                       reg_alpha=best_reg_alpha),
                               param_grid=param_test, scoring='roc_auc', iid=False, cv=5)
        gsearch.fit(self.X_train, self.y_train)
        logging.info("search reg_alpha......................")
        logging.info(gsearch.grid_scores_)
        logging.info(gsearch.best_params_)
        logging.info("score:{}".format(gsearch.best_score_))
        best_learning_rate = gsearch.best_params_['learning_rate']
        best_n_estimators = gsearch.best_params_['n_estimators']

        best_params = {"best_max_depth": best_max_depth,
                       "best_min_child_weight": best_min_child_weight,
                       "best_subsample": best_subsample,
                       "best_colsample_bytree": best_colsample_bytree,
                       "best_reg_alpha": best_reg_alpha,
                       "best_learning_rate": best_learning_rate,
                       "best_n_estimators": best_n_estimators,
                       "n_jobs": self.n_jobs
                       }
        json.dump(best_params, open("conf/{}".format(self.params_file), "w"))

        logging.info({"best_max_depth".format(best_max_depth)})
        logging.info("best_min_child_weight:{}".format(best_min_child_weight))
        logging.info("best_subsample:{}".format(best_subsample))
        logging.info("best_colsample_bytree:{}".format(best_colsample_bytree))
        logging.info("best_reg_alpha:{}".format(best_reg_alpha))
        logging.info("best_learning_rate:{}".format(best_learning_rate))
        logging.info("best_n_estimators:{}".format(best_subsample))
        logging.info("n_jobs:{}".format(self.n_jobs))

        self.best_estimator = gsearch.best_estimator_

        temp_model = gsearch.best_estimator_
        test_predprob = temp_model.predict_proba(self.X_test)[:, 1]
        logging.info("AUC Score (Test): %f" % metrics.roc_auc_score(self.y_test, test_predprob))

        # encode one-hot features
        grd_enc = OneHotEncoder()
        grd_enc.fit(temp_model.apply(self.X_train))
        self.grd_enc = grd_enc


    def save(self,
             model_file,
             encode_model_file
             ):
        joblib.dump(self.best_estimator, "model/{}".format(model_file))
        joblib.dump(self.grd_enc, "encode_model/{}".format(encode_model_file))

    def load(self,
             model_file,
             encode_model_file
             ):
        self.best_estimator = joblib.load("model/{}".format(model_file))
        self.grd_enc = joblib.load("encode_model/{}".format(encode_model_file))

    def predict_prob(self,
                     X):
        return self.best_estimator.predict_proba(X)

    def predict(self,
                X):
        return self.best_estimator.predict(X)

    def encode(self,
               X):
        return self.grd_enc.transform(X)

    def trainEvaluation(self):
        labels = self.y_test
        predicts = self.predict(self.X_test)
        self.evaluation(labels=labels,
                        predicts=predicts
                        )

    def evaluation(self, labels, predicts):
        assert (labels.shape == predicts.shape)
        m = labels.shape
        c11 = 0.0
        c10 = 0.0
        c01 = 0.0
        c00 = 0.0
        count0 = 0.0
        count1 = 0.0
        for idx in range(0, m[0]):
            if labels[idx] == 0:
                count0 += 1
            else:
                count1 += 1
            if labels[idx] == 0 and predicts[idx] == 0:
                c00 += 1
            elif labels[idx] == 0 and predicts[idx] == 1:
                c01 += 1
            elif labels[idx] == 1 and predicts[idx] == 0:
                c10 += 1
            else:
                c11 += 1
        logging.info("score: {}".format((c00 + c11) / float(c00 + c01 + c11 + c10 + 0.001)))
        logging.info("precision: {}".format(c11 / float(c11 + c01 + 0.001)))
        logging.info("recall: {}".format(c11 / float(c11 + c10 + 0.001)))
        logging.info("positive: {}, negative: {}".format(c01 + c11, c10 + c00))
        return c11, c10, c01, c00


class catboost(object):

    def __init__(self,
                 train_file,
                 params_file,
                 random_state,
                 test_size,
                 iterations,
                 depth,
                 rsm,
                 n_jobs
                 ):
        self.train_file = train_file
        self.params_file= params_file
        self.random_state = random_state
        self.test_size = test_size
        self.iterations = iterations
        self.depth = depth
        self.rsm = rsm
        self.n_jobs = n_jobs

    def process(self):
        X, y = load_svmlight_file(self.train_file)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            random_state=self.random_state,
                                                            test_size=self.test_size)
        self.X_train = X_train.toarray()
        self.y_train = y_train

        self.X_test = X_test.toarray()
        self.y_test = y_test

    def train(self,
            date):
        logging.basicConfig(filename="log/{}catBoostTrain.log".format(date), format="%(message)s",
                                filemode="a",
                                level=logging.INFO)
        param_test = {'iterations': self.iterations, 'depth': self.depth, 'learning_rate': [0.005, 0.01]}

        gsearch = GridSearchCV(
            estimator=CatBoostClassifier(loss_function='Logloss', verbose=True, l2_leaf_reg=3, thread_count=self.n_jobs),
            param_grid=param_test, scoring='roc_auc', iid=False, cv=2
            )

        gsearch.fit(self.X_train, self.y_train)
        logging.info("search iterations, depth and lr ......................")
        logging.info(gsearch.grid_scores_)
        logging.info(gsearch.best_params_)
        logging.info("score:{}".format(gsearch.best_score_))
        best_iterations = gsearch.best_params_["iterations"]
        best_depth = gsearch.best_params_["depth"]
        best_learning_rate = gsearch.best_params_["learning_rate"]

        param_test = {'rsm': self.rsm}
        gsearch = GridSearchCV(
            estimator=CatBoostClassifier(loss_function='Logloss', verbose=True, l2_leaf_reg=3,
                                         iterations=best_iterations, thread_count=self.n_jobs,
                                         depth=best_depth, learning_rate=best_learning_rate),
            param_grid=param_test, scoring='roc_auc', iid=False, cv=5
        )

        gsearch.fit(self.X_train, self.y_train)
        logging.info("search rsm ...................................")
        logging.info(gsearch.grid_scores_)
        logging.info(gsearch.best_params_)
        logging.info("score:{}".format(gsearch.best_score_))
        best_rsm = gsearch.best_params_["rsm"]

        best_params = {"best_iterations": best_iterations,
                       "best_depth": best_depth,
                       "best_learning_rate": best_learning_rate,
                       "best_rsm":best_rsm,
                       "n_jobs": self.n_jobs
                       }
        json.dump(best_params, open("conf/{}".format(self.params_file), "w"))

        self.best_estimator = gsearch.best_estimator_

        temp_model = gsearch.best_estimator_
        test_predprob = temp_model.predict_proba(self.X_test)[:, 1]
        logging.info("AUC Score (Test): %f" % metrics.roc_auc_score(self.y_test, test_predprob))


    def save(self,
             model_file
            ):
        joblib.dump(self.best_estimator, "model/{}".format(model_file))


    def load(self,
             model_file
            ):
        self.best_estimator = joblib.load("model/{}".format(model_file))


    def predict_prob(self,
                     X):
        return self.best_estimator.predict_proba(X)


    def predict(self,
                X):
        return self.best_estimator.predict(X)


    def trainEvaluation(self):
        labels = self.y_test
        predicts = self.predict(self.X_test)
        self.evaluation(labels=labels,
                        predicts=predicts
                        )


    def evaluation(self, labels, predicts):
        assert (labels.shape == predicts.shape)
        m = labels.shape
        c11 = 0.0
        c10 = 0.0
        c01 = 0.0
        c00 = 0.0
        count0 = 0.0
        count1 = 0.0
        for idx in range(0, m[0]):
            if labels[idx] == 0:
                count0 += 1
            else:
                count1 += 1
            if labels[idx] == 0 and predicts[idx] == 0:
                c00 += 1
            elif labels[idx] == 0 and predicts[idx] == 1:
                c01 += 1
            elif labels[idx] == 1 and predicts[idx] == 0:
                c10 += 1
            else:
                c11 += 1
        logging.info("score: {}".format((c00 + c11) / float(c00 + c01 + c11 + c10 + 0.001)))
        logging.info("precision: {}".format(c11 / float(c11 + c01 + 0.001)))
        logging.info("recall: {}".format(c11 / float(c11 + c10 + 0.001)))
        logging.info("positive: {}, negative: {}".format(c01 + c11, c10 + c00))
        return c11, c10, c01, c00
