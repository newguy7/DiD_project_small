import os
import sys
import pandas as pd

from did_project.components.psm import perform_psm
from did_project.components.did import perform_did

from did_project.exception.exception import DiDException
from did_project.logging.logger import logging

try:
    logging.info("Starting the DiD analysis pipeline...")    

    df = pd.read_csv("sample_data/synthetic_claims.csv")
    covariates = ['age', 'gender', 'risk_score', 'chronic_conditions', 'pre_er_visits']

    # Run PSM
    logging.info("Initiating PSM...")
    matched_df = perform_psm(df, covariates)
    matched_df.to_csv("outputs/matched_data.csv", index=False)

    # Run DiD
    logging.info("Initiating DiD analysis...")
    results = perform_did(matched_df)
    print(results)

    logging.info("Pipeline completed successfully.")

except Exception as e:    
    raise DiDException(e,sys)

