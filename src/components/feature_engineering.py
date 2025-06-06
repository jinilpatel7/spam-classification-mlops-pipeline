import os, sys
import pandas as pd

from src.logger import get_logger
from src.exception import CustomException

from sklearn.feature_extraction.text import TfidfVectorizer

logger = get_logger("feature_engineering")
def load_data(file_path: str) -> pd.DataFrame:
    try:
        logger.info("Feature Enginnering Started")
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.info("Data Loaded for FE")
        return df
    except Exception as e:
        logger.info("Unexpected error occurred while loading the preprocessed data.")
        raise CustomException(e, sys)
    
def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:

    try:
        logger.info("Applying TF-IDF Vectorization")
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values

        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.info("Vectorization Step Completed")
        return train_df, test_df
    except Exception as e:
        logger.info("Failed to apply TF-IDF")
        raise CustomException(e, sys)
    
def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        
    except Exception as e:
        raise CustomException(e, sys)
    
def main():
    try:
        max_features = 50

        train_data = load_data('./data/preprocess/train_processed.csv')
        test_data = load_data('./data/preprocess/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, os.path.join("./data", "processed_final", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed_final", "test_tfidf.csv"))
        logger.info("-----Data Saved-----")
    except Exception as e:
        logger.info("Failed to complete feature engineering process")
        raise CustomException(e, sys)
    
if __name__ == '__main__':
    main()
