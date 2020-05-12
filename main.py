import os

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

data_directory = "./data/"
data_list = os.listdir(data_directory)

for name in data_list:
    original = {
        'dt_mean': list(),
        'dt_std': list(),
        # 'svm_mean': list(),
        # 'svm_std': list(),
        'rf_mean': list(),
        'rf_std': list(),
        'knn_mean': list(),
        'knn_std': list(),
    }
    variance = {
        'dt_mean': list(),
        'dt_std': list(),
        # 'svm_mean': list(),
        # 'svm_std': list(),
        # 'svm_percent': list(),
        'rf_mean': list(),
        'rf_std': list(),
        'knn_mean': list(),
        'knn_std': list(),
        'dt_percent': list(),
        'rf_percent': list(),
        'knn_percent': list(),

    }
    chisquared = {
        'dt_mean': list(),
        'dt_std': list(),
        # 'svm_mean': list(),
        # 'svm_std': list(),
        'rf_mean': list(),
        'rf_std': list(),
        'knn_mean': list(),
        'knn_std': list(),
        'dt_percent': list(),
        'rf_percent': list(),
        'knn_percent': list(),
        # 'svm_percent': list(),
    }
    anova = {
        'dt_mean': list(),
        'dt_std': list(),
        # 'svm_mean': list(),
        # 'svm_std': list(),
        'rf_mean': list(),
        'rf_std': list(),
        'knn_mean': list(),
        'knn_std': list(),
        'dt_percent': list(),
        'rf_percent': list(),
        'knn_percent': list(),
        # 'svm_percent': list(),
    }
    principal = {
        'dt_mean': list(),
        'dt_std': list(),
        # 'svm_mean': list(),
        # 'svm_std': list(),
        'rf_mean': list(),
        'rf_std': list(),
        'knn_mean': list(),
        'knn_std': list(),
        'dt_percent': list(),
        'rf_percent': list(),
        'knn_percent': list(),
        # 'svm_percent': list(),
    }
    nruns = 30

    df = pd.read_csv("./data/" + name)

    X = df.drop(['id', 'Class'], axis=1)
    y = df['Class']

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    encoder = preprocessing.LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    dt = DecisionTreeClassifier()
    svm = SVC(kernel="rbf")
    rf = RandomForestClassifier()
    knn = KNeighborsClassifier()

    hp_dt = {"dt__min_samples_split": list(range(2, 51)), "dt__min_samples_leaf": list(range(2, 51)),
             "dt__max_depth": list(range(2, 31))}
    hp_svm = {"svm__C": list(range(1, 32769)), "svm__gamma": list(range(1, 32769))}
    hp_rf = {"rf__n_estimators": list(range(1, 1025)), "rf__max_depth": list(range(1, 21))}
    hp_knn = {"knn__n_neighbors": list(range(1, 51))}

    vt = VarianceThreshold()
    chi = SelectPercentile(chi2)
    f_cla = SelectPercentile(f_classif)
    pca = PCA(svd_solver='full')

    hp_vt = {"vt__threshold": [.8 * (1 - .8), .85 * (1 - .85), .9 * (1 - .9), .95 * (1 - .95)]}
    hp_chi = {"chi__percentile": [5, 10, 15, 20]}
    hp_f_cla = {"f_cla__percentile": [5, 10, 15, 20]}
    hp_pca = {"pca__n_components": [.8, .85, .9, .95]}

    # models = [dt, svm, rf, knn]
    # models2 = ['dt', 'svm', 'rf', 'knn']
    # hyperparameters = [hp_dt, hp_svm, hp_rf, hp_knn]
    models = [dt, rf, knn]
    models2 = ['dt', 'rf', 'knn']
    hyperparameters = [hp_dt, hp_rf, hp_knn]
    i = 0

    for m, h in zip(models, hyperparameters):
        stkf = StratifiedKFold(10)

        # pipe original
        pipe = Pipeline(steps=[(str(models2[i]), m)])
        search = RandomizedSearchCV(pipe, h, n_jobs=-1, cv=stkf, n_iter=nruns, scoring='balanced_accuracy')
        search.fit(X, y)
        # print(search.best_estimator_)
        original[models2[i] + '_mean'].append(search.cv_results_['mean_test_score'][search.best_index_])
        original[models2[i] + '_std'].append(search.cv_results_['std_test_score'][search.best_index_])

        # pipe vt
        try:
            pipe_vt = Pipeline(steps=[('vt', vt), (str(models2[i]), m)])
            search_vt = RandomizedSearchCV(pipe_vt, {**hp_vt, **h}, n_jobs=-1, cv=stkf, n_iter=nruns,
                                           scoring='balanced_accuracy')
            search_vt.fit(X, y)
            # print(search_vt.best_estimator_)
            variance[models2[i] + '_mean'].append(search_vt.cv_results_['mean_test_score'][search_vt.best_index_])
            variance[models2[i] + '_std'].append(search_vt.cv_results_['std_test_score'][search_vt.best_index_])
            variance[models2[i] + '_percent'].append(
                (X.shape[1] - sum(search_vt.best_estimator_.named_steps['vt'].get_support())) / (
                        X.shape[1] + sum(search_vt.best_estimator_.named_steps['vt'].get_support())))
        except:
            pipe_vt = Pipeline(steps=[('vt', vt), (str(models2[i]), m)])
            search_vt = RandomizedSearchCV(pipe_vt, {**h}, n_jobs=-1, cv=stkf, n_iter=nruns,
                                           scoring='balanced_accuracy')
            search_vt.fit(X, y)
            # print(search_vt.best_estimator_)
            variance[models2[i] + '_mean'].append(search_vt.cv_results_['mean_test_score'][search_vt.best_index_])
            variance[models2[i] + '_std'].append(search_vt.cv_results_['std_test_score'][search_vt.best_index_])
            variance[models2[i] + '_percent'].append(
                (X.shape[1] - sum(search_vt.best_estimator_.named_steps['vt'].get_support())) / (
                        X.shape[1] + sum(search_vt.best_estimator_.named_steps['vt'].get_support())))


        # pipe chi
        pipe_chi = Pipeline(steps=[('chi', chi), (str(models2[i]), m)])
        search_chi = RandomizedSearchCV(pipe_chi, {**hp_chi, **h}, n_jobs=-1, cv=stkf, n_iter=nruns,
                                        scoring='balanced_accuracy')
        search_chi.fit(X, y)
        chisquared[models2[i] + '_mean'].append(search_chi.cv_results_['mean_test_score'][search_chi.best_index_])
        chisquared[models2[i] + '_std'].append(search_chi.cv_results_['std_test_score'][search_chi.best_index_])
        chisquared[models2[i] + '_percent'].append(
            (X.shape[1] - sum(search_chi.best_estimator_.named_steps['chi'].get_support())) / (
                    X.shape[1] + sum(search_chi.best_estimator_.named_steps['chi'].get_support())))

        # pipe f_classif
        pipe_f_cla = Pipeline(steps=[('f_cla', f_cla), (str(models2[i]), m)])
        search_f_cla = RandomizedSearchCV(pipe_f_cla, {**hp_f_cla, **h}, n_jobs=-1, cv=stkf, n_iter=nruns,
                                          scoring='balanced_accuracy')
        search_f_cla.fit(X, y)
        anova[models2[i] + '_mean'].append(search_f_cla.cv_results_['mean_test_score'][search_f_cla.best_index_])
        anova[models2[i] + '_std'].append(search_f_cla.cv_results_['std_test_score'][search_f_cla.best_index_])
        anova[models2[i] + '_percent'].append(
            (X.shape[1] - sum(search_f_cla.best_estimator_.named_steps['f_cla'].get_support())) / (
                    X.shape[1] + sum(search_f_cla.best_estimator_.named_steps['f_cla'].get_support())))

        # pipe pca
        pipe_pca = Pipeline(steps=[('pca', pca), (str(models2[i]), m)])
        search_pca = RandomizedSearchCV(pipe_pca, {**hp_pca, **h}, n_jobs=-1, cv=stkf, n_iter=nruns,
                                        scoring='balanced_accuracy')
        search_pca.fit(X, y)
        principal[models2[i] + '_mean'].append(search_pca.cv_results_['mean_test_score'][search_pca.best_index_])
        principal[models2[i] + '_std'].append(search_pca.cv_results_['std_test_score'][search_pca.best_index_])
        principal[models2[i] + '_percent'].append(
            (X.shape[1] - search_pca.best_estimator_.named_steps['pca'].n_components_) / (
                    X.shape[1] + search_pca.best_estimator_.named_steps['pca'].n_components_))

        i += 1
    pd.DataFrame(original).to_csv("./results/" + str(name.split('.')[0]) + "_original.csv", index=False)
    pd.DataFrame(variance).to_csv("./results/" + str(name.split('.')[0]) + "_vt.csv", index=False)
    pd.DataFrame(chisquared).to_csv("./results/" + str(name.split('.')[0]) + "_chi.csv", index=False)
    pd.DataFrame(anova).to_csv("./results/" + str(name.split('.')[0]) + "_f_cla.csv", index=False)
    pd.DataFrame(principal).to_csv("./results/" + str(name.split('.')[0]) + "_pca.csv", index=False)
