# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text
import seaborn as sns


import statistics


clusters_df = pd.read_csv(r'D:\XBG Gitlab\ML_Rockfall\Training_data_labeled_v5_equal_size.csv')

# Identify input and target columns
input_cols, target_col = clusters_df.columns[0:-1], clusters_df.columns[-1]
inputs_df, targets = clusters_df[input_cols].copy(), clusters_df[target_col].copy()
# Identify numeric and categorical columns
numeric_cols = clusters_df[input_cols].select_dtypes(include=np.number).columns.tolist()
categorical_cols = clusters_df[input_cols].select_dtypes(include='object').columns.tolist()

def test(depth, estimators):
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs_df[numeric_cols],
                                                                            targets,
                                                                            test_size=0.25)

    # Impute and scale numeric columns
    imputer = SimpleImputer().fit(train_inputs)
    train_inputs = imputer.transform(train_inputs)
    scaler = MinMaxScaler().fit(train_inputs)
    train_inputs = scaler.transform(train_inputs)

    imputer = SimpleImputer().fit(val_inputs)
    val_inputs1 = imputer.transform(val_inputs)
    scaler = MinMaxScaler().fit(val_inputs1)
    val_inputs1 = scaler.transform(val_inputs1)

    model = RandomForestClassifier()
    model.fit(train_inputs, train_targets)
    score = model.score(val_inputs1, val_targets)

    pred = model.predict(val_inputs1)
    result = val_inputs.assign(label=pred)
    result = result.rename(columns={'label':'prediction'})
    check = pd.concat([result, val_targets], axis=1)
    check["compare"] = check.prediction*2+check.label

    check.to_csv(r'D:\ML_Granada_v2\Result_ML\equal_size3.csv', columns=['//X','Y', 'Z', 'compare'], index=False)

    importance_df = pd.DataFrame({
        'feature': inputs_df.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_df.head(10)

    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(val_targets, pred, normalize='true')

    pred1 = model.predict(train_inputs)
    matrix1 = confusion_matrix(train_targets, pred1, normalize='true')

    return score, matrix, model, importance_df, pred1, matrix1

values = []
for i in range (1):
    score, matrix, model, importance_df, pred1, matrix1 = test(1, 100)
    values.append(score)
    print(statistics.mean(values))
    print(score)
    print(matrix1)
    plt.figure(1)
    sns.heatmap(matrix1, annot=False, cmap="RdYlGn")
    plt.show()

    plt.figure(figsize=(9,5))
    plt.title('Feature Importance')
    sns.barplot(data=importance_df.head(10), x='importance', y='feature')
    plt.show()

#plt.figure(figsize=(80,20))
#plot_tree(model.estimators_[5], feature_names=inputs_df.columns, filled=True, rounded=True, class_names=str(model.classes_));
#plt.show()

importance_df.head(10)