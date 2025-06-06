import os, sys
from src.logger import get_logger
from src.exception import CustomException

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

logger = get_logger("data_preprocessing")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def transform_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.split()  # Simpler tokenization
    text = [word for word in text if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def preprocess_df(df,text_column='text', target_column='target'):
    try:
        logger.info("Data perprocessing Started")
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])

        logger.info("Target Column Encoded")
        df = df.drop_duplicates(keep='first')

        logger.info("Duplicates Removed")
        df.loc[:, text_column] = df[text_column].apply(transform_text)

        logger.info("Text Column Transformed")
        return df
    except Exception as e:
        logger.info("Error in Preprocess_df")
        raise CustomException(e, sys)
    
def main(text_column='text', target_column='target'):
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.info("Train & Test Data Loaded")

        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        data_path = os.path.join("./data", "preprocess")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.info("-----Processed Data Saved-----")
    except Exception as e:
        logger.info("Error in data perprocessing.")
        raise CustomException(e, sys)
    
if __name__ == '__main__':
    main()