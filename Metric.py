from abc import ABC, abstractmethod
import numpy as np
from sklearn import metrics

class Metric(ABC):
    @abstractmethod
    def compute_value(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        '''
        :param y_true: true label 
        :param y_pred: predicted label
        :return: the metric value
        '''
        pass

    @abstractmethod
    def get_name(self) -> str:
        '''
        :return: the metric name for logging purposes
        '''
        pass

class MSE(Metric):
    def compute_value(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return metrics.mean_squared_error(y_true, y_pred)
    
    def get_name(self) -> str:
        return "MSE"

class R2(Metric):
    def compute_value(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return metrics.r2_score(y_true, y_pred)
    
    def get_name(self) -> str:
        return "R2"

class Accuracy(Metric):
    def compute_value(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return metrics.accuracy_score(y_true, y_pred)
    
    def get_name(self) -> str:
        return "Accuracy"

class ROCAUC(Metric):
    def compute_value(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return metrics.roc_auc_score(y_true, y_pred)
    
    def get_name(self) -> str:
        return "ROC AUC"
