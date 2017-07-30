#coding=utf-8
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import logging
from time import strftime, gmtime
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
import json

class gdbt_model(object):
    def __init__(self, train_file, random_state, test_size, n_estimators, max_depth, min_samples_split, max_features, min_samples_leaf, subsample, params_file, n_jobs):
        self.time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        logging.basicConfig(filename="log/{}.log".format(self.time), format="%(message)s",
                            filemode="a",
                            level=logging.INFO)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state, test_size=self.test_size)
        self.X_train = X_train.toarray()
        self.y_train = y_train

        self.X_test = X_test.toarray()
        self.y_test = y_test


    def train(self):
        param_test = {'n_estimators': self.n_estimators, 'learning_rate': [0.005, 0.01, 0.05, 0.1]}
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
                       "best_max_depth":best_max_depth,
                       "best_min_samples_split":best_min_samples_split,
                       "best_subsample":best_subsample,
                       "n_jobs":self.n_jobs
                       }
        json.dump(best_params, open("conf/{}".format(self.params_file), "w"))

        logging.info({"best_learning_rate".format(best_learning_rate)})
        logging.info("best_n_estimators:{}".format(best_n_estimators))
        logging.info("best_max_features:{}".format(best_max_features))
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

    def save(self, model_file):
        joblib.dump(self.best_estimator, model_file)
        joblib.dump(self.grd_enc, "encode_{}".format(model_file))

    def load(self, model_file):
        self.best_estimator = joblib.load(model_file)
        self.grd_enc = joblib.load("encode_{}".format(model_file))

    def predict_prob(self, X):
        return self.best_estimator.predict_proba(X)[:,1]

    def encode(self, X):
        return self.grd_enc.transform(X)