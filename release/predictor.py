from __future__ import print_function
from __future__ import division
import numpy as np

from joblib import Parallel
from sklearn import metrics
from rdkit import Chem
from rdkit.Chem import Descriptors
from catboost import Pool, CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor as RFR


from utils import get_fp, get_desc, normalize_desc, cross_validation_split

PROPERTY_PREDICTORS = {'mwt': Descriptors.ExactMolWt}
PROPERTY_ESTIMATORS = {'RFR': RFR, 'CatBoost': CatBoostClassifier}




class VanillaQSAR(object):
    def __init__(self, model_class='RFR', model_instance=None, model_params=None,
                 model_type='classifier', ensemble_size=5, normalization=False):
        super(VanillaQSAR, self).__init__()
        self.model_instance = model_instance
        self.model_params = model_params
        self.model_class = model_class

        self.ensemble_size = ensemble_size
        self.model = []
        self.normalization = normalization
        if model_type not in ['classifier', 'regressor']:
            raise InvalidArgumentError("model type must be either"
                                       "classifier or regressor")
        self.model_type = model_type
        if isinstance(self.model_instance, list):
            assert(len(self.model_instance) == self.ensemble_size)
            assert(isinstance(self.model_params, list))
            assert(len(self.model_params) == self.ensemble_size)
            for i in range(self.ensemble_size):
                self.model.append(self.model_instance[i](**model_params[i]))
        else:
            for _ in range(self.ensemble_size):
                self.model.append(self.model_instance(**model_params))
        if self.normalization:
            self.desc_mean = [0]*self.ensemble_size
        self.metrics_type = None

    def fit_model(self, data, cv_split='stratified'):

        if self.model_class =='RFR' or self.model_class == '...':

            eval_metrics = []
            x = data.x
            if self.model_type == 'classifier' and data.binary_y is not None:
                y = data.binary_y
            else:
                y = data.y
            cross_val_data, cross_val_labels = cross_validation_split(x=x, y=y,
                                                                      split=cv_split,
                                                                      n_folds=self.ensemble_size)
            for i in range(self.ensemble_size):
                train_x = np.concatenate(cross_val_data[:i] +
                                         cross_val_data[(i + 1):])
                test_x = cross_val_data[i]
                train_y = np.concatenate(cross_val_labels[:i] +
                                         cross_val_labels[(i + 1):])
                test_y = cross_val_labels[i]
                if self.normalization:
                    train_x, desc_mean = normalize_desc(train_x)
                    self.desc_mean[i] = desc_mean
                    test_x, _ = normalize_desc(test_x, desc_mean)
                self.model[i].fit(train_x, train_y.ravel())
                predicted = self.model[i].predict(test_x)
                if self.model_type == 'classifier':
                    eval_metrics.append(metrics.f1_score(test_y, predicted))
                    self.metrics_type = 'F1 score'
                elif self.model_type == 'regressor':
                    r2 = metrics.r2_score(test_y, predicted)
                    eval_metrics.append(r2)
                    self.metrics_type = 'R^2 score'
                else:
                    raise RuntimeError()
            return eval_metrics, self.metrics_type

        elif self.model_class =='CatBoost':

            eval_metrics = []
            x = data.x
            if self.model_type == 'classifier' and data.binary_y is not None:
                y = data.binary_y
            else:
                y = data.y
            cross_val_data, cross_val_labels = cross_validation_split(x=x, y=y,
                                                                      split=cv_split,
                                                                      n_folds=self.ensemble_size)
            for i in range(self.ensemble_size):
                train_x = np.concatenate(cross_val_data[:i] +
                                         cross_val_data[(i + 1):])
                test_x = cross_val_data[i]
                train_y = np.concatenate(cross_val_labels[:i] +
                                         cross_val_labels[(i + 1):])
                test_y = cross_val_labels[i]
                if self.normalization:
                    train_x, desc_mean = normalize_desc(train_x)
                    self.desc_mean[i] = desc_mean
                    test_x, _ = normalize_desc(test_x, desc_mean)

                train_pool = Pool(train_x, train_y.ravel())
                test_pool = Pool(test_x, test_y.ravel())

                self.model[i].fit(train_pool)
                predicted = self.model[i].predict(test_pool)
                if self.model_type == 'classifier':
                    eval_metrics.append(metrics.f1_score(test_y, predicted))
                    self.metrics_type = 'F1 score'
                elif self.model_type == 'regressor':
                    r2 = metrics.r2_score(test_y, predicted)
                    eval_metrics.append(r2)
                    self.metrics_type = 'R^2 score'
                else:
                    raise RuntimeError()
            return eval_metrics, self.metrics_type

    def i_th_model_predict(self, i, x):

        m = self.model[i]
        if self.normalization:
            x, _ = normalize_desc(x, self.desc_mean[i])

            x = Pool(x)
        return m.predict(x)


    def load_model(self, path):
        # TODO: add iterable path object instead of static path
        self.model = []
        for i in range(self.ensemble_size):
            m = joblib.load(path + str(i) + '.pkl')
            self.model.append(m)
        if self.normalization:
            arr = np.load(path + 'desc_mean.npy')
            self.desc_mean = arr

    def save_model(self, path):
        assert self.ensemble_size == len(self.model)
        for i in range(self.ensemble_size):
            joblib.dump(self.model[i], path + str(i) + '.pkl')
        if self.normalization:
            np.save(path + 'desc_mean.npy', self.desc_mean)

    def predict(self, objects=None, average=True, get_features=None,
                **kwargs):

        if self.model_class == 'RFR' or self.model_class == '...':

            objects = np.array(objects)
            invalid_objects = []
            processed_objects = []
            if get_features is not None:
                x, processed_indices, invalid_indices = get_features(objects,
                                                                     **kwargs)
                processed_objects = objects[processed_indices]
                invalid_objects = objects[invalid_indices]
            else:
                x = objects
            if len(x) == 0:
                processed_objects = []
                prediction = []
                invalid_objects = objects
            else:
                prediction = []
                for i in range(self.ensemble_size):
                    m = self.model[i]
                    if self.normalization:
                        x, _ = normalize_desc(x, self.desc_mean[i])
                    prediction.append(m.predict(x))
                prediction = np.array(prediction)
                if average:
                    if self.model_type == 'classifier':
                        # unique, counts = np.unique(prediction, return_counts=True)
                        # prediction = np.array([sorted(tuple(zip(unique, counts)), key=lambda x: x[1], reverse=True)[0][0]])
                        prediction = np.round(prediction.mean(axis=0))
                    else:
                        prediction = prediction.mean(axis=0)

        elif self.model_class =='CatBoost':

            objects = np.array(objects)
            invalid_objects = []
            processed_objects = []
            if get_features is not None:
                x, processed_indices, invalid_indices = get_features(objects,
                                                                     **kwargs)
                processed_objects = objects[processed_indices]
                invalid_objects = objects[invalid_indices]
            else:
                x = objects
            if len(x) == 0:
                processed_objects = []
                prediction = []
                invalid_objects = objects
            else:
                prediction = Parallel(n_jobs=self.ensemble_size, prefer="threads")(
                    self.i_th_model_predict(i, x) for i in range(self.ensemble_size))

                prediction = np.array(prediction)
                if average:
                    if self.model_type == 'classifier':

                        prediction = np.round(prediction.mean(axis=0))

                    else:
                        prediction = prediction.mean(axis=0)

        return processed_objects, prediction, invalid_objects


class PropertyPredictor():
    def __init__(self, rdkit_desc, model_type):
        self.desc = rdkit_desc
        self.model_type = model_type

    def predict(self, smiles, **kwargs):
        valid_smiles = []
        props = []
        nan_smiles = []
        for sm in smiles:
            if isinstance(sm, str):
                m = Chem.MolFromSmiles(sm)
                if not m:
                    nan_smiles.append(sm)
                    continue
            else:
                m = sm
            valid_smiles.append(sm)
            props.append(self.desc(m))
        props = np.array(props)

        return valid_smiles, props, nan_smiles

