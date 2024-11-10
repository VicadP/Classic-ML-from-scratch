from typing import List
import logging
import numpy as np
import pandas as pd
from Metrics import Metric
from sklearn.model_selection import train_test_split

logger = logging.getLogger('model evaluation')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S %p')

class ModelEvaluation():
    '''
    Осуществляет разбиение данных и производит оценку модели на основании типа проблемы и передаваемых метрик
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray, problem_type: str = "regression", 
                 test_size: float = 0.3, shuffle: bool = True, random_state: int = 42,
                 metrics = List[Metric]):
        '''
        :param X: матрица данных
        :param y: целевая переменная
        :param problem_type: тип решаемой проблемы, возможные значения "regression" и "classification". 
                             Если указан тип "classification", то расчитываются как soft(вероятности), так и hard(метки) предсказания.
        :param test_size: доля данных, откладываемых для тестовой выборки
        :param shuffle: перемешать ли данные при разбиении
        :param random_state: ядро рандомайзера
        :param metrics: массив метрик для оценки модели
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
        )
        assert problem_type in ["regression", "classification"], "Некорректный ввод. Допустимые значения 'regression' и 'classification'"
        self.problem_type = problem_type
        self.metrics = metrics
        self.results = []

    def evaluate_model(self, model, model_name = "DefaultName"):
        '''
        Обучает и оценивает модель, сохраняя результаты

        :param model: инстанс модели
        :param model_name: имя модели, используется для сохранения результата
        '''
        #logger.info(f"Обучаем модель {model}")
        model.fit(self.X_train, self.y_train)
        if self.problem_type == "regression":
            y_pred = model.predict(self.X_test)
        else:
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)
            if y_pred_proba.ndim == 2: # для обработки предсказаний sklearn, т.к. там возвращаются вероятности для обоих классов
                y_pred_proba = y_pred_proba[:, 1]
        metrics_dict = {}
        for metric in self.metrics:
            if metric.get_name() == "ROC AUC":
                metric_value = metric.compute_value(self.y_test, y_pred_proba)
            else:
                metric_value = metric.compute_value(self.y_test, y_pred)
            metrics_dict[metric.get_name()] = metric_value
            #logger.info(f"{model_name}: {metric.get_name()} = {metric_value:.3f}")
        self.results.append({"model": model_name, **metrics_dict})
    
    def get_result(self) -> pd.DataFrame:
        '''
        :return: результат оценки модели
        '''
        return pd.DataFrame(self.results)