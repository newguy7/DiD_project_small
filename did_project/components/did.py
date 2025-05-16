import os
import sys
import pandas as pd
import statsmodels.formula.api as smf
from did_project.exception.exception import DiDException
from did_project.logging.logger import logging


def perform_did(df):
    try:
        logging.info("Structuring panel data for DiD analysis...")
        panel_df = pd.concat([
            df.assign(period='pre', er_visits=df['pre_er_visits']),
            df.assign(period='post', er_visits=df['post_er_visits'])
        ])

        panel_df['post'] = (panel_df['period'] == 'post').astype(int)
        panel_df['did'] = panel_df['post'] * panel_df['treated']

        logging.info("Fitting DiD regression model...")
        model = smf.ols('er_visits ~ treated + post + did', data=panel_df).fit()

        logging.info("DiD analysis complete.")
        return model.summary()

    except Exception as e:        
        raise DiDException(e,sys)
    
    
