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

clusters_df = pd.read_csv(r'D:\ML_Granada_v2\Training_data_labeled_v2.csv')

# Identify input and target columns
input_cols, target_col = clusters_df.columns[0:-2], clusters_df.columns[-2]
inputs_df, targets = clusters_df[input_cols].copy(), clusters_df[target_col].copy()
# Identify numeric and categorical columns
numeric_cols = clusters_df[input_cols].select_dtypes(include=np.number).columns.tolist()
categorical_cols = clusters_df[input_cols].select_dtypes(include='object').columns.tolist()

inputs_df, targets = clusters_df[numeric_cols].copy(), clusters_df[target_col].copy()


# Impute and scale numeric columns
imputer = SimpleImputer().fit(inputs_df[numeric_cols])
inputs_df[numeric_cols] = imputer.transform(inputs_df[numeric_cols])
scaler = MinMaxScaler().fit(inputs_df[numeric_cols])
inputs_df[numeric_cols] = scaler.transform(inputs_df[numeric_cols])

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_jobs=-1, random_state=42)

model.fit(inputs_df, targets)

model.score(inputs_df, targets)

importance_df = pd.DataFrame({
    'feature': inputs_df.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

importance_df.head(10)
