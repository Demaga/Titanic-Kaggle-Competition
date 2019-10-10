#import sys 
from tabulate import tabulate
import pandas as pd
import time
import operator
import math
#import matplotlib
import numpy as np 
#import scipy as sp 
#import IPython
#from IPython import display
import random
import time
#from subprocess import check_output
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

from copy import deepcopy

# Visualization
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


def clean(data):
    #print("Количество пассажиров 1 класса:", data.loc[(data.Pclass == 1), "Survived"]#.count())
    #print("Количество пассажиров 1 класса с неизвестным возрастом:", data.loc[#(data.Pclass == 1) & (data.Age.isna()), "Survived"].count())
    #print("Количество детей 1 класса:", data.loc[(data.Pclass == 1) & (data.Age < 12), #"Survived"].count())
    #print("Количество детей 2 класса:", data.loc[(data.Pclass == 2) & (data.Age < 12), #"Survived"].count())
    #print("Количество детей 3 класса:", data.loc[(data.Pclass == 3) & (data.Age < 12), #"Survived"].count())

    first_class_mean = data.loc[(data.Pclass == 1) & (data.Age > 12), "Age"].mean()

    male_first_class_age_mean = data.loc[(data.Sex == "male") & (data.Pclass == 1) & (data.Age > 12), "Age"].mean()
    female_first_class_age_mean = data.loc[(data.Sex == "female") & (data.Pclass == 1) & (data.Age > 12), "Age"].mean()

    male_second_class_age_mean = data.loc[(data.Sex == "male") & (data.Pclass == 2) & (data.Age > 12), "Age"].mean()
    female_second_class_age_mean = data.loc[(data.Sex == "female") & (data.Pclass == 2), "Age"].mean()

    male_third_class_age_mean = data.loc[(data.Sex == "male") & (data.Pclass == 3), "Age"].mean()
    female_third_class_age_mean = data.loc[(data.Sex == "female") & (data.Pclass == 3) , "Age"].mean()

    #print("\nMale age 1 class mean:",male_first_class_age_mean, "\nFemale age 1 class #mean:", female_first_class_age_mean)
    #print("\nMale age 2 class mean:",male_second_class_age_mean, "\nFemale age 2 class #mean:", female_second_class_age_mean)
    #print("\nMale age 3 class mean:",male_third_class_age_mean, "\nFemale age 3 class #mean:", female_third_class_age_mean)
    #print()
    #print(first_class_mean)
    #print("************************************")

    data.loc[(data.Pclass == 1) & (data.Age.isna() == True) & (data.Sex == "male"), "Age"] = male_first_class_age_mean
    data.loc[(data.Pclass == 1) & (data.Age.isna() == True) & (data.Sex == "female"), "Age"] = female_first_class_age_mean
    data.loc[(data.Pclass == 2) & (data.Age.isna() == True) & (data.Sex == "male"), "Age"] = male_first_class_age_mean
    data.loc[(data.Pclass == 2) & (data.Age.isna() == True) & (data.Sex == "female"), "Age"] = female_first_class_age_mean
    data.loc[(data.Pclass == 3) & (data.Age.isna() == True) & (data.Sex == "male"), "Age"] = male_first_class_age_mean
    data.loc[(data.Pclass == 3) & (data.Age.isna() == True) & (data.Sex == "female"), "Age"] = female_first_class_age_mean
    
    data.Embarked.fillna("C", inplace=True)

    data.drop(columns=["PassengerId", "Cabin", "Ticket", "Name"], inplace=True)

    data["FamilySize"] = data["SibSp"] + data["ParCh"] + 1
    data["IsAlone"] = 1 # True
    data.loc[data["FamilySize"] > 1, "IsAlone"] = 0 # False
    data["FareBin"] = pd.qcut(data["Fare"], 6)
    data["AgeBin"] = pd.qcut(data["Age"], 7)

    label = LabelEncoder()
    data["Sex_Code"] = label.fit_transform(data["Sex"])
    data["Embarked_Code"] = label.fit_transform(data["Embarked"])
    data["AgeBin_Code"] = label.fit_transform(data["AgeBin"])
    data["FareBin_Code"] = label.fit_transform(data["FareBin"])



    return data


def meta_parameter_selection_algorithm(model, parameter, left, mid, right, train_x_dummy, train_y_dummy, test_x_dummy, test_y_dummy):
    row_index = 0
    accuracy = 0

    alg_left = deepcopy(model)
    alg_mid = deepcopy(model)
    alg_right = deepcopy(model)

    setattr(alg_left, parameter, left)
    setattr(alg_mid, parameter, mid)
    setattr(alg_right, parameter, right)

    alg_left.fit(train_x_dummy, train_y_dummy)
    alg_mid.fit(train_x_dummy, train_y_dummy)
    alg_right.fit(train_x_dummy, train_y_dummy) 

    y_left_pred = alg_left.predict(test_x_dummy)
    y_mid_pred = alg_mid.predict(test_x_dummy)
    y_right_pred = alg_right.predict(test_x_dummy)

    false_positive_rate, true_positive_rate, thresholds = roc_curve (test_y_dummy, y_left_pred)
    roc_auc_left = auc(false_positive_rate, true_positive_rate)
    false_positive_rate, true_positive_rate, thresholds = roc_curve (test_y_dummy, y_mid_pred)
    roc_auc_mid = auc(false_positive_rate, true_positive_rate)
    false_positive_rate, true_positive_rate, thresholds = roc_curve (test_y_dummy, y_right_pred)
    roc_auc_right = auc(false_positive_rate, true_positive_rate)

    print("\nLeft: "+str(parameter)+ "= "+ str(getattr(alg_left,parameter)) +"; accuracy = " + str(roc_auc_left) +";")
    print("Middle: "+str(parameter)+ "= "+ str(getattr(alg_mid,parameter)) +"; accuracy = " + str(roc_auc_mid) +";")
    print("Right: "+str(parameter)+ "= "+ str(getattr(alg_right,parameter)) +"; accuracy = " + str(roc_auc_right) +";")
    print("********************************************************")
    
    accuracies = dict()
    accuracies[str(getattr(alg_left,parameter))] = roc_auc_left 
    accuracies[str(getattr(alg_mid,parameter))] = roc_auc_mid 
    accuracies[str(getattr(alg_right,parameter))] = roc_auc_right 

    while row_index < 5:
        row_index = row_index + 1
        if (type(left) == int):
            if (roc_auc_left > roc_auc_mid):
                left = left
                right = math.ceil(right - (right - mid)/2)
                mid = math.ceil((left + right) / 2)
            elif (roc_auc_right > roc_auc_mid):
                right = right
                left = math.ceil(left - (left - mid)/2)
                mid = math.ceil((left + right) / 2)
            else:
                left = math.ceil((left + mid) / 2)
                right = math.ceil((right + mid) / 2)
                mid = mid
        else:
            if (roc_auc_left > roc_auc_mid):
                left = left
                right = right - (right - mid)/2
                mid = (left + right) / 2
            elif (roc_auc_right > roc_auc_mid):
                right = right
                left = left - (left - mid)/2
                mid = (left + right) / 2
            else:
                left =(left + mid) / 2
                right = (right + mid) / 2
                mid = mid

        alg_left = deepcopy(model)
        alg_mid = deepcopy(model)
        alg_right = deepcopy(model)

        setattr(alg_left, parameter, left)
        setattr(alg_mid, parameter, mid)
        setattr(alg_right, parameter, right)

        alg_left.fit(train_x_dummy, train_y_dummy)
        alg_mid.fit(train_x_dummy, train_y_dummy)
        alg_right.fit(train_x_dummy, train_y_dummy) 

        y_left_pred = alg_left.predict(test_x_dummy)
        y_mid_pred = alg_mid.predict(test_x_dummy)
        y_right_pred = alg_right.predict(test_x_dummy)

        false_positive_rate, true_positive_rate, thresholds = roc_curve (test_y_dummy, y_left_pred)
        roc_auc_left = auc(false_positive_rate, true_positive_rate)
        false_positive_rate, true_positive_rate, thresholds = roc_curve (test_y_dummy, y_mid_pred)
        roc_auc_mid = auc(false_positive_rate, true_positive_rate)
        false_positive_rate, true_positive_rate, thresholds = roc_curve (test_y_dummy, y_right_pred)
        roc_auc_right = auc(false_positive_rate, true_positive_rate)

        print("\nLeft: "+str(parameter)+ "= "+ str(getattr(alg_left,parameter)) +"; accuracy = " + str   (roc_auc_left) +";")
        print("Middle: "+str(parameter)+ "= "+ str(getattr(alg_mid,parameter)) +"; accuracy = " + str    (roc_auc_mid) +";")
        print("Right: "+str(parameter)+ "= "+ str(getattr(alg_right,parameter)) +"; accuracy = " + str   (roc_auc_right) +";")
        print("********************************************************")

        accuracies[str(getattr(alg_left,parameter))] = roc_auc_left 
        accuracies[str(getattr(alg_mid,parameter))] = roc_auc_mid 
        accuracies[str(getattr(alg_right,parameter))] = roc_auc_right 
    
    print("\nBest "+str(parameter)+" = " + str(max(accuracies.items(), key=operator.itemgetter(1))[0]) + "\nBest accuracy = " + str(accuracies[max(accuracies.items(), key=operator.itemgetter(1))[0]]))

    return max(accuracies.items(), key=operator.itemgetter(1))[0]


def parameters_selection_algorithm(data):
    Target = ["Survived"]

    # define x variables
    data_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
    data_xy_bin = Target + data_x_bin

    data_dummy = pd.get_dummies(data[data_x_bin])
    data_x_dummy = data_dummy.columns.tolist()
    data_xy_dummy = Target + data_x_dummy

    scaler = MinMaxScaler()
    for x in data_x_dummy:
        if data_dummy[x].dtype == 'float64':
            data_dummy[x] = scaler.fit_transform(data_dummy[x])

    y = data.Survived.values.squeeze()

    train_x_dummy, test_x_dummy, train_y_dummy, test_y_dummy = model_selection.train_test_split(data_dummy[data_x_dummy], data[Target], random_state = 0)
    
    models = [
        ensemble.GradientBoostingClassifier(),
        ensemble.ExtraTreesClassifier(),

        linear_model.LogisticRegressionCV(),

        naive_bayes.BernoulliNB(),

        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        svm.LinearSVC(max_iter=10000),

        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),

        XGBClassifier()
    ]

    possible_attributes = {
        "learning_rate": {
            "left": 0.01,
            "mid": 0.1,
            "right": 0.2,
        },
        "max_leaf_nodes": {
            "left": 25,
            "mid": 100,
            "right": 175,
        },
        "n_estimators": {
            "left": 50,
            "mid": 100,
            "right": 150,
        },
        "min_split_loss": {
            "left": 0,
            "mid": 10,
            "right": 25,
        },
        "min_child_weight": {
            "left": 0,
            "mid": 15,
            "right": 35,
        },
        "min_samples_leaf": {
            "left": 5,
            "mid": 15,
            "right": 35,
        },
        "max_features": {
            "left": 15,
            "mid": 30,
            "right": 45,
        },
        "min_impurity_decrease": {
            "left": 0.01,
            "mid": 0.1,
            "right": 0.2,            
        }

    }
    

    #create table to compare MLA metrics
    MLA_columns = ['MLA Name', "learning-rate", "max_leaf_nodes",  "n_estimators", "min_split_loss", "min_child_weight", 'MLA Test Accuracy Mean', 'MLA Time']
    MLA_compare = pd.DataFrame(columns = MLA_columns)

    #create table to compare MLA predictions
    MLA_predict = data[Target]

    k = 0
    row_index = 0
    while k < 4:
        random_order = list()
        for i in range(10):
            r=random.randint(0, 4)
            if r not in random_order:
                 random_order.append(r)

        for model in models:
            print("---------------------------------\n---------------------------------")
            print(type(model).__name__)
            model_attributes = dir(model)
            i = 0
            start = time.time()
            while i < 4:
                MLA_compare.loc[row_index, "MLA Name"] = type(model).__name__ + str(i)
                attribute = list(possible_attributes)[i]
                if attribute in model_attributes:
                    parameter = meta_parameter_selection_algorithm(model, attribute,    possible_attributes[attribute]["left"], possible_attributes[attribute] ["mid"], possible_attributes[attribute]["right"], train_x_dummy,     train_y_dummy, test_x_dummy, test_y_dummy)
                    if attribute == "learning_rate":
                        parameter = float(parameter)
                    else:
                        parameter = int(parameter)
                    MLA_compare.loc[row_index, parameter] = parameter
                    setattr(model, attribute, parameter)
                i = i + 1
            
            model.fit(train_x_dummy, train_y_dummy)
            y = model.predict(test_x_dummy)
            false_positive_rate, true_positive_rate, thresholds = roc_curve (test_y_dummy, y)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            print(model)

            end = time.time()

            MLA_compare.loc[row_index, "MLA Test Accuracy Mean"] = roc_auc
            MLA_compare.loc[row_index, "MLA Time"] = end - start

            print("\nTotal result = ", roc_auc)
        
            row_index = row_index + 1

        
        k = k + 1

    MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
    print("************************************")
    print(tabulate(MLA_compare[["MLA Name", "MLA Test Accuracy Mean", "MLA Time"]], headers='keys', tablefmt='psql'))

    
data = pd.read_csv("data/train.csv")
data = clean(data)
parameters_selection_algorithm(data)

