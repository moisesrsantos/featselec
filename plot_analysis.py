import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare


perf = pd.read_csv("./results/performance.csv")

plt.figure(1)
sns.boxplot(data = perf)
plt.show()

stat1, p1 = friedmanchisquare(perf["Chi-squared"],perf["F ANOVA"],perf["PCA"],perf["Variance"],perf["Original"])
print("Performance: ")
print('Statistics=%.3f, p=%.3f' % (stat1, p1))
alpha = 0.05
if p1 > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')



red = pd.read_csv("./results/reduction.csv")

plt.figure(2)
sns.barplot(data = red, ci="sd", orient = "h")
plt.show()

print("Reduction: ")
stat2, p2 = friedmanchisquare(perf["Chi-squared"],perf["F ANOVA"],perf["PCA"],perf["Variance"])
print('Statistics=%.3f, p=%.3f' % (stat2, p2))
alpha = 0.05
if p2 > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')



time = pd.read_csv("./results/time.csv")

plt.figure(3)
sns.violinplot(data = time)
plt.show()

print("Time to Fit: ")
stat3, p3 = friedmanchisquare(time["Chi-squared"],time["F ANOVA"],time["PCA"],time["Variance"], time["Original"])
print('Statistics=%.3f, p=%.3f' % (stat3, p3))
alpha = 0.05
if p3 > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
