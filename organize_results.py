import os
import pandas as pd

results_directory = "./results/"
results_list = os.listdir(results_directory)

original = {
    'mean': list(),
    'std': list(),
    'time': list()
}
variance = {
    'mean': list(),
    'std': list(),
    'percent': list(),
    'time': list()
}
chisquared = {
    'mean': list(),
    'std': list(),
    'percent': list(),
    'time': list()
}
anova = {
    'mean': list(),
    'std': list(),
    'percent': list(),
    'time': list()
}
principal = {
    'mean': list(),
    'std': list(),
    'percent': list(),
    'time': list()
}


for name in results_list:
    df = pd.read_csv("./results/" + name)
    if("chi" in name):
        chisquared['mean'].append(df["dt_mean"].loc[0])
        chisquared['mean'].append(df["rf_mean"].loc[0])
        chisquared['mean'].append(df["svm_mean"].loc[0])
        chisquared['mean'].append(df["knn_mean"].loc[0])

        chisquared['std'].append(df["dt_std"].loc[0])
        chisquared['std'].append(df["rf_std"].loc[0])
        chisquared['std'].append(df["svm_std"].loc[0])
        chisquared['std'].append(df["knn_std"].loc[0])

        chisquared['percent'].append(df["dt_percent"].loc[0])
        chisquared['percent'].append(df["rf_percent"].loc[0])
        chisquared['percent'].append(df["svm_percent"].loc[0])
        chisquared['percent'].append(df["knn_percent"].loc[0])

        chisquared['time'].append(df["dt_time"].loc[0])
        chisquared['time'].append(df["rf_time"].loc[0])
        chisquared['time'].append(df["svm_time"].loc[0])
        chisquared['time'].append(df["knn_time"].loc[0])
    elif("f_cla" in name):
        anova['mean'].append(df["dt_mean"].loc[0])
        anova['mean'].append(df["svm_mean"].loc[0])
        anova['mean'].append(df["rf_mean"].loc[0])
        anova['mean'].append(df["knn_mean"].loc[0])

        anova['std'].append(df["dt_std"].loc[0])
        anova['std'].append(df["rf_std"].loc[0])
        anova['std'].append(df["svm_std"].loc[0])
        anova['std'].append(df["knn_std"].loc[0])

        anova['percent'].append(df["dt_percent"].loc[0])
        anova['percent'].append(df["rf_percent"].loc[0])
        anova['percent'].append(df["svm_percent"].loc[0])
        anova['percent'].append(df["knn_percent"].loc[0])

        anova['time'].append(df["dt_time"].loc[0])
        anova['time'].append(df["rf_time"].loc[0])
        anova['time'].append(df["svm_time"].loc[0])
        anova['time'].append(df["knn_time"].loc[0])

    elif("original" in name):
        original['mean'].append(df["dt_mean"].loc[0])
        original['mean'].append(df["svm_mean"].loc[0])
        original['mean'].append(df["rf_mean"].loc[0])
        original['mean'].append(df["knn_mean"].loc[0])

        original['std'].append(df["dt_std"].loc[0])
        original['std'].append(df["rf_std"].loc[0])
        original['std'].append(df["svm_std"].loc[0])
        original['std'].append(df["knn_std"].loc[0])

        original['time'].append(df["dt_time"].loc[0])
        original['time'].append(df["rf_time"].loc[0])
        original['time'].append(df["svm_time"].loc[0])
        original['time'].append(df["knn_time"].loc[0])

    elif("pca" in name):
        principal['mean'].append(df["dt_mean"].loc[0])
        principal['mean'].append(df["svm_mean"].loc[0])
        principal['mean'].append(df["rf_mean"].loc[0])
        principal['mean'].append(df["knn_mean"].loc[0])

        principal['std'].append(df["dt_std"].loc[0])
        principal['std'].append(df["rf_std"].loc[0])
        principal['std'].append(df["svm_std"].loc[0])
        principal['std'].append(df["knn_std"].loc[0])

        principal['percent'].append(df["dt_percent"].loc[0])
        principal['percent'].append(df["rf_percent"].loc[0])
        principal['percent'].append(df["svm_percent"].loc[0])
        principal['percent'].append(df["knn_percent"].loc[0])

        principal['time'].append(df["dt_time"].loc[0])
        principal['time'].append(df["rf_time"].loc[0])
        principal['time'].append(df["svmtime"].loc[0])
        principal['time'].append(df["knn_time"].loc[0])

    elif("vt" in name):
        variance['mean'].append(df["dt_mean"].loc[0])
        variance['mean'].append(df["svm_mean"].loc[0])
        variance['mean'].append(df["rf_mean"].loc[0])
        variance['mean'].append(df["knn_mean"].loc[0])

        variance['std'].append(df["dt_std"].loc[0])
        variance['std'].append(df["rf_std"].loc[0])
        variance['std'].append(df["svm_std"].loc[0])
        variance['std'].append(df["knn_std"].loc[0])

        variance['percent'].append(df["dt_percent"].loc[0])
        variance['percent'].append(df["rf_percent"].loc[0])
        variance['percent'].append(df["svm_percent"].loc[0])
        variance['percent'].append(df["knn_percent"].loc[0])

        variance['time'].append(df["dt_time"].loc[0])
        variance['time'].append(df["rf_time"].loc[0])
        variance['time'].append(df["svm_time"].loc[0])
        variance['time'].append(df["knn_time"].loc[0])


pd.DataFrame({"Chi-squared": chisquared['mean'],"F ANOVA": anova['mean'], "PCA": principal['mean'], "Variance": variance['mean'], "Original": original['mean']}).to_csv("./results/performance.csv",index = False)
pd.DataFrame({"Chi-squared": chisquared['percent'],"F ANOVA": anova['percent'], "PCA": principal['percent'], "Variance": variance['percent']}).to_csv("./results/reduction.csv",index = False)
pd.DataFrame({"Chi-squared": chisquared['time'],"F ANOVA": anova['time'], "PCA": principal['time'], "Variance": variance['time']}).to_csv("./results/time.csv",index = False)

