import rpy2.robjects as robjects
import numpy as np
import pandas as pd


class Observation():
    def __init__(self):
        self.statValues = {}
        self.modelName = ""

    def setModelName(self, nameOfModel):
        self.modelName = nameOfModel

    def addStatMetric(self, metricName, metricValue):
        self.statValues[metricName] = metricValue


def to_str(val):
    return str(val).split('"')[1]


def flatten_dict(d, prefix='__'):
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for sub_key, sub_value in flatten_dict(value).items():
                    yield sub_key, sub_value
            else:
                yield key, value

    return dict(items())


def evalSingleModel(X, y_test, clf, modelName, variant):
    y_pred = clf.predict(X)
    print('SCORING number of target: ' + str(np.sum(y_pred)))
    print('real number of target==1: ' + str(np.sum(y_test)))

    # send the data to R
    groundTruth = robjects.IntVector(y_test)
    y_predicted = robjects.IntVector(y_pred)

    r = robjects.r
    r.source("rEvaluation.R")

    evaluateAllTheThings = robjects.globalenv['evaluateAllTheThings']
    res = evaluateAllTheThings(groundTruth, y_predicted)
    statsResults = dict([[to_str(j.names), j[0]] for i, j in enumerate(res)])
    obs = Observation()
    obs.setModelName(modelName + '-' + variant)

    for _kpi, value in statsResults.items():
        obs.addStatMetric(_kpi, value)

    obs.addStatMetric('typeOfRun', variant)

    return obs


def niceDisplayOfResults(predictionResults):
    results = []
    for res in predictionResults:
        results.append(res.__dict__)

    l = list(map(flatten_dict, results))
    results = pd.DataFrame.from_dict(l)

    validation_res = results[results.typeOfRun == 'validation'][
        ['modelName', 'kappa', 'Error']]
    validation_res.modelName = validation_res.modelName.str.split('_').str[0]
    validation_res = validation_res.sort_values('kappa', ascending=False)

    train_res = results[results.typeOfRun != 'validation']
    overview = train_res.groupby([train_res.modelName.str.split('_').str[0]]).describe().unstack(
        fill_value=0).loc[:,
               pd.IndexSlice[:, ['mean', 'std']]][['kappa', 'Error']]
    overview.columns = ['{0[0]}_{0[1]}'.format(tup) for tup in overview.columns]
    overview.sort_values('kappa_mean', ascending=False)

    return validation_res, overview,
