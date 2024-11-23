import logging
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.distance import euclidean
from typing import Callable
from typing import Tuple

logger = logging.getLogger('model training')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S %p')

class KNNEstimator(ABC):
    '''
    Интерфейс для метода ближайших соседей для классификации и регрессии.
    Метрика для дистанции - L2.
    '''

    def __init__(self, n_neighbors: int = 5, metric: Callable = euclidean, weights: str = "uniform"):
        '''
        :param n_neighbors: кол-во ближайших соседей
        :param metric: метрика дистанции между наблюдениями
        :param weights: стратегия взвешивания ближайших соседей
        '''
        self.n_neighbors = n_neighbors
        self.metric = metric
        assert weights in ["uniform", "distance"], "Некорректный ввод, допустимые значения 'uniform', 'distance'"
        self.weights = weights
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        :param X: матрица независимых переменных
        :param y: зависимая переменная
        '''
        self.X = X.copy()
        self.y = y.copy()

    def apply_weights(self, distances: np.ndarray, neighbors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Расчитывает веса для n_neighbors ближайщих соседей

        :param distances: дистанция до n_neighbors ближайщих соседей
        :param neighbors: зависимая переменная n_neighbors ближайщих соседей
        :return: массив весов и массив лейблов
        '''
        if self.weights == 'uniform':
            weights_by_neighbor = np.squeeze(np.array([(1, neighbor) for _, neighbor in zip(distances, neighbors)]))
            return weights_by_neighbor[:, 0], weights_by_neighbor[:, 1]
        if self.weights == 'distance':
            weights_by_neighbor = np.squeeze(np.array([((1 / distance + 1e-4), neighbor) for distance, neighbor in zip(distances, neighbors)]))
            return weights_by_neighbor[:, 0], weights_by_neighbor[:, 1]

    def compute_prediction(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Расчитывает дистанцию для единичного наблюдения до всех наблюдений в обучающей выборке, выделяя n_neighbors ближайших соседей.
        Триггерит расчет весов для найденых ближайших соседей.

        :param observation: вектор из независимых переменных (единичное наблюдение)
        :return: массив весов и массив лейблов
        '''
        n_neighbors = np.squeeze(np.array(sorted([(self.metric(x, observation), y) for x, y in zip(self.X, self.y)])[:self.n_neighbors]))
        n_weights, n_labels = self.apply_weights(n_neighbors[:, 0], n_neighbors[:, 1])
        return n_weights, n_labels

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

class MyKNNRegressor(KNNEstimator):
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X:
            weights, labels = self.compute_prediction(x)
            predictions.append(np.sum(weights * labels) / np.sum(weights))
        return np.array(predictions)
    

class MyKNNClassifiers(KNNEstimator):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        '''
        :param X: матрица независимых переменных
        :return: вектор предсказаний-вероятностей
        '''
        predictions = []
        for x in X:
            weights, labels = self.compute_prediction(x)
            predictions.append(np.sum(weights[labels == 1]) / np.sum(weights))
        return np.array(predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)