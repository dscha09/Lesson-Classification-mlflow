import pandas as pd
import pickle
import seaborn as sns
import mlflow
import sys
import time, os, fnmatch, shutil
import json

##from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split
from fastai.text import *
from sklearn.metrics import precision_score, recall_score, accuracy_score
from datetime import *
from DataFunctions import ElasticFunctions as ef
from fastai.text import load_learner
from pathlib import Path
from fastai.text import load_learner
    

# MLflow Tracking functions
def log_params(param_name, value):
    print(f"[INFO] Logging parameter: {param_name}")
    mlflow.log_param(param_name, value)
    
def log_metric(metric_name, value, step=None):
    print(f"[INFO] Logging metric: {metric_name}")
    mlflow.log_metric(metric_name, value, step=None)
    
def log_artifact(source_dir, artifact_path=None):
    print(f"[INFO] Logging artifacts in {source_dir}...")
    mlflow.log_artifacts(local_dir, artifact_path=None)
    

# Evaluation
def plot_confusion_matrix(y_actual, y_pred):
    data = {
            'y_Actual': y_actual,
            'y_Predicted': y_pred
    }
    
    print(f"Precision={precision_score(y_actual,y_pred, average='weighted')}")
    print(f"Recall={recall_score(y_actual,y_pred, average='weighted')}")
    print(f"Accuracy={accuracy_score(y_actual, y_pred)}")

    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], margins = True)
    confusion_matrix = confusion_matrix.iloc[:-1,:-1]

    sns.heatmap(confusion_matrix, annot=True)


def get_credentials(filename):
    """Load credentials from JSON file
    """
    with open(filename) as f:
        data = json.load(f)
    return data


def get_for_predict_dataframe():

    """credentials = {
        # "ip_and_port": "52.163.240.214:9200",
        "ip_and_port": "52.230.8.63:9200",
        "username": "elastic",
        "password": "Welcometoerni!"
    }

    prodCredentials = {
        "ip_and_port": "52.163.240.214:9200",
        # "ip_and_port": "52.230.8.63:9200",
        "username": "elastic",
        "password": "Welcometoerni!"
    }"""

    credentials = get_credentials(args.credentials)

    # Get lessons data from database
    df = ef.getSentences(credentials)

    to_predict_par_df = df[["isLesson", "paragraph"]].replace(True, int(1)).replace(False, int(0))

    return to_predict_par_df

def setup_mlflow():
    print("MLflow Version:", mlflow.version.VERSION)
    print("Tracking URI:", mlflow.tracking.get_tracking_uri())

    experiment_name = "lesson-classif-windows"
    print("experiment_name:",experiment_name)
    mlflow.set_experiment(experiment_name)

    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    print("experiment_id:",experiment_id)

    now = int(time.time()+.5)


def main():

    setup_mlflow()

    # Get dataset from Elasticsearch
    to_predict_par_df = get_for_predict_dataframe()

    #ROOT_PATH = r"C:\Users\Test Machine\Documents\ADB-CognitiveSearch-ML\pipeline\functions\models"
    ROOT_PATH = "./models"

    # Load saved model file
    #learn = load_learner(Path(ROOT_PATH), "lesson_classif-04-05-2020_11-05-30_PM.pkl")
    lesson_learner = load_learner(Path(ROOT_PATH), args.model_filename)

    forecasts = []
    actual = to_predict_pardf.isLesson.values

    for p in to_predict_par_df.paragraph: 
        ##print(learn_classif.predict(p))
        forecasts.append(try_int(lesson_learner.predict(p)[0]))
    ##plot_confusion_matrix(actual, forecasts)

    # Get sentences
    credentials = get_credentials(args.credentials)
    df2 = ef.getSentences(credentials)

    # Update isLessons in sentences
    to_predict_par_df2 = to_predict_par_df
    to_predict_par_df2.isLesson = forecasts

    to_predict_par_df2.isLesson = to_predict_par_df2.isLsson.replace(int(1), True).replace(int(0), False)
    
    df2.isLesson, df2.paragraph = to_predict_par_df2.isLesson, to_predict_par_df2.paragraph
        
    ##ef.updateSentences(credentials, df2)
    print(df2.head())
    

if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--model_file", dest="model_file", default=None, required=True, action='store_true')
    parser.add_argument("--credentials_file", dest="credentials_file", default=None, required=True, action='store_true')
    args = parser.parse_args()
    main()
    
