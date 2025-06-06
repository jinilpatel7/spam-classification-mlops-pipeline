import os, sys
import pandas as pd

from src.logger import get_logger
from src.exception import CustomException

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

logger = get_logger("model_building")
def load_data(file_path: str) -> pd.DataFrame:
    try:
        logger.info("Loading Data for Model Training")
        df = pd.read_csv(file_path)
        logger.info("Data Loaded for Training")
        return df
    except Exception as e:
        logger.info("Unexpected error occurred while loading the processed data.")
        raise CustomException(e, sys)
    
def train_model(X_train: np.ndarray, y_train: np.ndarray, param: dict) -> RandomForestClassifier:
    try:
        logger.info("Model Training Started ....")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be same")
        
        clf = RandomForestClassifier(n_estimators=param['n_estimators'], random_state=param['random_state'])

        clf.fit(X_train, y_train)
        logger.info("Model Training Completed")
        return clf
    except Exception as e:
        raise CustomException(e, sys)
    
def save_model(model, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        logger.info("Unexpected error occurred while saving model")
        raise CustomException(e, sys)
    
def main():
    try:
        params = {'n_estimators':25, 'random_state':2 }

        train_data = load_data('./data/processed_final/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)

        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

        logger.info("-----Trained Model Saved-----")
    except Exception as e:
        logger.info("Unexpected error occurred while building model")
        raise CustomException(e, sys)
    
if __name__ == '__main__':
    main()