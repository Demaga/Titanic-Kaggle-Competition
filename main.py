#import sys 
from tabulate import tabulate
import pandas as pd
#import matplotlib
#import numpy as np 
#import scipy as sp 
#import IPython
#from IPython import display
#import random
#import time
#from subprocess import check_output
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Visualization
#import plotly
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import matplotlib.pylab as pylab
#import seaborn as sns
#from pandas.plotting import scatter_matrix

# Visualization
#mpl.style.use('ggplot')
#sns.set_style('white')
#pylab.rcParams['figure.figsize'] = 12,8
print("Libraries loaded!\n************************************")

data = pd.read_csv("data/train.csv")
print(data.head())
print("************************************")
print(data.info())
print("************************************")
print(data.isnull().sum())

first_class_mean = data[data.Pclass == 1].Age.mean()
second_class_mean = data[data.Pclass == 2].Age.mean()
third_class_mean = data[data.Pclass == 3].Age.mean()
print("************************************")
print("First class mean: ",first_class_mean)
print("Second class mean: ", second_class_mean)
print("Third class mean: ", third_class_mean)
print("************************************")


data.loc[(data.Pclass == 1) & pd.isna(data.Age), "Age"] = first_class_mean
data.loc[(data.Pclass == 2) & pd.isna(data.Age), "Age"] = second_class_mean
data.loc[(data.Pclass == 3) & pd.isna(data.Age), "Age"] = third_class_mean

data.Embarked.fillna("C", inplace=True)

data.drop(columns=["PassengerId", "Cabin", "Ticket", "Name"], inplace=True)

print(data.isnull().sum())
print("************************************")

# unite data to categories
data["FamilySize"] = data["SibSp"] + data["ParCh"] + 1
data["IsAlone"] = 1 # True
data.loc[data["FamilySize"] > 1, "IsAlone"] = 0 # False
data["FareBin"] = pd.qcut(data["Fare"], 4)
data["AgeBin"] = pd.qcut(data["Age"], 5)

#print(data.loc[data.Sex == "female"].loc[data.Pclass == 1].loc[data.IsAlone == 1])
print(data["Fare"].describe())

print(data.iloc[:15, :15])
print("************************************")

# convert objects to categories 
label = LabelEncoder()
data["Sex_Code"] = label.fit_transform(data["Sex"])
data["Embarked_Code"] = label.fit_transform(data["Embarked"])
data["AgeBin_Code"] = label.fit_transform(data["AgeBin"])
data["FareBin_Code"] = label.fit_transform(data["FareBin"])

print("***************************")
print(data.head())

Target = ["Survived"]

# define x variables
data_x = ['Sex','Pclass', 'Embarked', 'SibSp', 'ParCh', 'Age', 'Fare', 'FamilySize', 'IsAlone']
data_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'SibSp', 'ParCh', 'Age', 'Fare']
data_xy =  Target + data_x

data_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data_xy_bin = Target + data_x_bin

data_dummy = pd.get_dummies(data[data_x])
data_x_dummy = data_dummy.columns.tolist()
data_xy_dummy = Target + data_x_dummy
print(data_dummy.head())
y = data.Survived.values.squeeze()

print('Dummy X Y: ', data_xy_dummy, '\n') 
# show correlation info
for x in data_x:
    if data[x].dtype != 'float64' :
        print('Correlation:', x)
        print(data[[x, Target[0]]].groupby(x, as_index=False).mean())
        print("************************************", "\n")


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(max_leaf_nodes=candidate_max_leaf_nodes[0]),
    ensemble.ExtraTreesClassifier(max_leaf_nodes=candidate_max_leaf_nodes[1]),
    ensemble.ExtraTreesClassifier(max_leaf_nodes=candidate_max_leaf_nodes[2]),
    ensemble.ExtraTreesClassifier(max_leaf_nodes=candidate_max_leaf_nodes[3]),
    ensemble.ExtraTreesClassifier(max_leaf_nodes=candidate_max_leaf_nodes[4]),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(max_iter=10000),
    
    #Trees
    tree.DecisionTreeClassifier(max_leaf_nodes=candidate_max_leaf_nodes[0]),
    tree.DecisionTreeClassifier(max_leaf_nodes=candidate_max_leaf_nodes[1]),
    tree.DecisionTreeClassifier(max_leaf_nodes=candidate_max_leaf_nodes[2]),
    tree.DecisionTreeClassifier(max_leaf_nodes=candidate_max_leaf_nodes[3]),
    tree.DecisionTreeClassifier(max_leaf_nodes=candidate_max_leaf_nodes[4]),
    tree.DecisionTreeClassifier(max_leaf_nodes=candidate_max_leaf_nodes[5]),
    tree.ExtraTreeClassifier(max_leaf_nodes=candidate_max_leaf_nodes[0]),
    tree.ExtraTreeClassifier(max_leaf_nodes=candidate_max_leaf_nodes[1]),
    tree.ExtraTreeClassifier(max_leaf_nodes=candidate_max_leaf_nodes[2]),
    tree.ExtraTreeClassifier(max_leaf_nodes=candidate_max_leaf_nodes[3]),
    tree.ExtraTreeClassifier(max_leaf_nodes=candidate_max_leaf_nodes[4]),
    tree.ExtraTreeClassifier(max_leaf_nodes=candidate_max_leaf_nodes[5]),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
]

train_x_dummy, test_x_dummy, train_y_dummy, test_y_dummy = model_selection.train_test_split(data_dummy[data_x_dummy], data[Target], random_state = 0)

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = data[Target]

#index through MLA and save performance to table
row_index = 0
i = 0
ii = 0
iii = 0
for alg in MLA:
    print(alg)
    print("-----------------------------------\n")

    #set name and parameters
    MLA_name = alg.__class__.__name__
    if (MLA_name == "DecisionTreeClassifier"):
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name + str(candidate_max_leaf_nodes[i])
        i = i + 1
    elif (MLA_name == "ExtraTreeClassifier"):
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name + str(candidate_max_leaf_nodes[ii])
        ii = ii + 1
    elif (MLA_name == "ExtraTreesClassifier"):
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name + str(candidate_max_leaf_nodes[iii])
        iii = iii + 1
    else:
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, data[data_x_bin], data[Target], cv  = cv_split) 

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    #MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    #save MLA predictions
    alg.fit(data[data_x_bin], data[Target])
    MLA_predict[MLA_name] = alg.predict(data[data_x_bin])
    
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
print("************************************")
print(tabulate(MLA_compare[["MLA Name", "MLA Test Accuracy Mean", "MLA Time"]], headers='keys', tablefmt='psql'))

# best accuracy: 81.6 %
# work is still in progress


