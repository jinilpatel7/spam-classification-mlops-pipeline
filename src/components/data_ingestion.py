import os, sys
import pandas as pd

from src.logger import get_logger
from src.exception import CustomException


from sklearn.model_selection import train_test_split

logger = get_logger("data_ingestion")
def load_data(data_url: str) -> pd.DataFrame:
    """Load That from CSV file."""
    logger.info("Data Ingestion started")
    try:
        df = pd.read_csv(data_url)
        logger.info("Data Loaded")
        return df
    except Exception as e:
        logger.info("Unexpected error occurred while loading the data.")
        raise CustomException(e, sys)
    
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Data Preprocess code"""
    logger.info("Basic Data Preprocessing Started")
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.info("Basic Data Preprocessing Completed")
        return df
    
    except Exception as e:
        logger.info("Error in data perprocessing")
        raise CustomException(e, sys)
    
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save Train and Test data"""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.info("Train & Test Data saved")
    except Exception as e:
        logger.info("Unexpected error occurred while saving the data")
        raise CustomException(e, sys)
    
def main():
    try:
        test_size = 0.2 

        data_path = "https://raw.githubusercontent.com/jinilpatel7/spam-classification-mlops-pipeline/refs/heads/main/Experiment_Notebook/spam.csv"
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')

    except Exception as e:
        logger.info("Failed to complete data ingestion process")
        raise CustomException(e, sys)

if __name__ == '__main__':
    main()