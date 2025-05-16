import os
import sys
import pandas as pd
from did_project.exception.exception import DiDException
from did_project.logging.logger import logging

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def perform_psm(df, covariates, treatment_col='treated', score_col='propensity_score'):
    try:
        if not all(col in df.columns for col in covariates):
            raise DiDException(f"One or more covariates are missing from the input dataframe.")

        logging.info("Fitting logistic regression for propensity scores...")
        X = df[covariates]
        y = df[treatment_col]

        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X,y)

        df[score_col] = model.predict_proba(X)[:,1]

        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

        if treated.empty or control.empty:
            raise DiDException("Insufficient data in treated or control group for matching.")

        logging.info("Performing nearest neighbor matching...")
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(control[[score_col]])
        _, indices = nn.kneighbors(treated[[score_col]])

        matched_control = control.iloc[indices.flatten()].copy()

        matched_df = pd.concat([treated, matched_control])
        logging.info(f"Matched dataset size: {matched_df.shape}")

        return matched_df
    
    except Exception as e:        
        raise DiDException(e,sys)
    
       
