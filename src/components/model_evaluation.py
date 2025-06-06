import os, sys

from src.logger import get_logger
from src.exception import CustomException

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

logger = get_logger("model_evaluation")
def load_model(file_path: str):
    try:
        logger.info("Loading Trained Model")
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.info("Model Loaded")
        return model
    except Exception as e:
        logger.info("Unexpected error occurred while loading the Model.")
        raise CustomException(e, sys)
    

def load_data(file_path: str) -> pd.DataFrame:
    try:
        logger.info("Loading Data for model evaluation")
        df = pd.read_csv(file_path)
        logger.info("Data Loaded")
        return df
    except Exception as e:
        logger.info("Unexpected error occurred while loading the data")
        raise CustomException(e, sys)
    
def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        logger.info("Model Evaluation Started....")
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        metrics_dict = {
            'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'auc' : auc
        }
        return metrics_dict
    except Exception as e:
        logger.info("Unexpected error occurred while Evaluating Model")
        raise CustomException(e, sys)
    
def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        logger.info("Unexpected error occurred while Saving metrics")
        raise CustomException(e, sys)
    
def main():
    try:
        clf = load_model('./models/model.pkl')

        test_data = load_data('./data/processed_final/test_tfidf.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)

        save_metrics(metrics, 'reports/metrics.json')
        logger.info("-----Model Evaluation Completed and Saved the report-----")
    except Exception as e:
        logger.info("Unexpected error occurred while Model Evaluation process")
        raise CustomException(e, sys)
    
if __name__ == '__main__':
    main()