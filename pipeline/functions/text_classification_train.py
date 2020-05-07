import pandas as pd
import pickle
import seaborn as sns
import mlflow
import sys
import time, os, fnmatch, shutil

##from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split
from fastai.text import *
from sklearn.metrics import precision_score, recall_score, accuracy_score
from datetime import *
from DataFunctions import ElasticFunctions as ef


# Arguments
parser = ArgumentParser()
parser.add_argument("--data_dir", dest="data_dir", default="../data", required=True, action='store_true')
args = parser.parse_args()


class UlmFit: 
    def __init__(self, data_dir, train_data_file, test_data_file): 
        self.data_lm =  TextLMDataBunch.from_csv(data_dir, train_data_file)
        self.data_classifier = TextClasDataBunch.from_csv(data_dir, train_data_file, 
                                                          vocab=self.data_lm.train_ds.vocab, 
                                                          bs=20)
        self.lm_learner = language_model_learner(self.data_lm, AWD_LSTM, drop_mult=0.5)
        self.lesson_learner = text_classifier_learner(self.data_classifier, 
                                                      AWD_LSTM, drop_mult=0.5, 
                                                      metrics=[accuracy, Precision(), Recall()]).to_fp16()
        
    def train_language_model(self):
        self.lm_learner.lr_find()
        self.lm_learner.recorder.plot(suggestion=True)
        min_grad_lr = self.lm_learner.recorder.min_grad_lr
        self.lm_learner.fit_one_cycle(5, min_grad_lr)
    
    def save_language_model(self):
        self.lm_learner.save_encoder('language_model')
    
    def save_lm_file(self):
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        self.lm_learner.export(f"../models/lm-{timestamp}.pkl")
        return "../models/lm-{timestamp}.pkl"
        
    def save_classifier_file(self):
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        self.lesson_learner.export(f"../models/lesson_classif-{timestamp}.pkl")
        return "../models/lesson_classif-{timestamp}.pkl"
    
    def train_text_classification(self):
        self.lesson_learner.load_encoder('language_model')
        self.lesson_learner.lr_find()
        self.lesson_learner.recorder.plot(suggestion=True)
        min_grad_lr = self.lesson_learner.recorder.min_grad_lr
        self.lesson_learner.fit_one_cycle(50, min_grad_lr)
    

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


def main():
    # UlmFit
    ulmfit_obj = UlmFit(data_dir, "train_lesson_classif.csv", "test_lesson_classif.csv")
    
    #  Language model
    print("[INFO] Training language model...")
    ulmfit_obj.train_language_model()
    ulmfit_obj.save_language_model()
    lm_path = ulmfit_obj.save_lm_file()
    print(f"[INFO] Language model saved in {lm_path}")

    # Lesson classifier
    print("[INFO] Training lesson classification model...")
    ulmfit_obj.train_text_classification()
    lesson_classif_path = ulmfit_obj.save_lm_file()
    print(f"[INFO] Lesson classification model saved in {lesson_classif_path}")

    ulmfit_obj.lesson_learner.recorder.plot_losses()


if __name__ == "__main__":
    main()
    
    


    
