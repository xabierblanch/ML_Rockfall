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

    model = GradientBoostingClassifier()
    model.fit(train_inputs, train_targets)
    score = model.score(val_inputs1, val_targets)

    pred = model.predict(val_inputs1)
    result = val_inputs.assign(label=pred)
    result = result.rename(columns={'label':'prediction'})
    check = pd.concat([result, val_targets], axis=1)
    check["compare"] = check.prediction*2+check.label

    check.to_csv(r'D:\ML_Granada_v2\Result_ML\check.csv', columns=['//X','Y', 'Z', 'compare'], index=False)

    importance_df = pd.DataFrame({
        'feature': inputs_df.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_df.head(10)

    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(val_targets, pred, normalize='true')

    return score, matrix

values = []
for i in range (100):
    score, matrix = test(1, 100)
    values.append(score)
    print(statistics.mean(values))
    #print(score)
    #print(matrix)
