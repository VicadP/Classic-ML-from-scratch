import logging
from typing import DefaultDict
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

logger = logging.getLogger('model training')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S %p')

class RandomForestEstimator(ABC):
    '''
    Интерфейс для случайного леса для регрессии и классификации с OOB оценкой
    '''

    def __init__(self, n_estimators: int = 100, max_depth: int | None = None, 
                 min_samples_split: int = 2, max_features: str = "sqrt", criterion: str | None = None):
        '''
        :param n_estimators: кол-во деревьев в ансамбле
        :param max_depth: глубина дерева
        :param min_samples_split: минимальное кол-во наблюдений в ноде для сплита
        :param max_features: кол-во признаков, выбираемых для построения дерева
        :param criterion: мера чистоты
        '''
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self._decision_trees = []
        self.oob_predictions = defaultdict(list)
        self.oob_score = None

    @abstractmethod
    def _init_decision_tree(self):
        '''
        :return: дерево решений
        '''
        pass

    @abstractmethod
    def _compute_oob_score(self, y_true: np.ndarray, oob_predictions: DefaultDict) -> float:
        '''
        :param y_true: зависимая переменная
        :param oob_predictions: среднее для регрессии и мажоритарный класс для классификации на oob выборке
        :return: коэффициент детерминации
        '''
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        :param X: матрица независимых переменных
        :param y: зависимая переменная
        '''
        self.n, self.p = X.shape
        row_idx = list(range(self.n))
        for b in range(self.n_estimators):
            sample_row_idx = np.random.choice(row_idx, size=self.n, replace=True)
            X_sampled, y_sampled = X[sample_row_idx, :], y[sample_row_idx]
            decision_tree = self._init_decision_tree()
            decision_tree.fit(X_sampled, y_sampled)
            self._decision_trees.append(decision_tree)
            oob_row_idx = list(set(row_idx) - set(sample_row_idx))
            if oob_row_idx:
                oob_prediction = decision_tree.predict(X[oob_row_idx])
                for idx, prediction in zip(oob_row_idx, oob_prediction):
                    self.oob_predictions[idx].append(prediction)
        self.oob_score = self._compute_oob_score(y, self.oob_predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        :param X: матрица независимых переменных
        :return: вектор предсказаний
        '''
        tree_predictions = []
        for decision_tree in self._decision_trees:
            prediction = decision_tree.predict(X).reshape(-1,1)
            tree_predictions.append(prediction)
        tree_predictions = np.concatenate(tree_predictions, axis=1)
        return self._compute_ensemble_prediction(tree_predictions)

    @abstractmethod
    def _compute_ensemble_prediction(self, tree_predictions: np.ndarray) -> np.ndarray:
        '''
        :return среднее или мажоритарный класс по n_estimators предсказаниям
        '''
        pass

class MyRandomForestRegressor(RandomForestEstimator):
    def _init_decision_tree(self):
        assert self.criterion is not None, "Необходимо задать меру чистоты"
        return DecisionTreeRegressor(max_depth = self.max_depth,
                                     min_samples_split = self.min_samples_split,
                                     max_features = self.max_features,
                                     criterion = self.criterion)

    def _compute_oob_score(self, y_true: np.ndarray, oob_predictions: DefaultDict) -> float:
        true_labels = np.array([y_true[idx] for idx in oob_predictions.keys()])
        oob_prediction = np.array([np.mean(predictions) for predictions in oob_predictions.values()])
        return r2_score(true_labels, oob_prediction)

    def _compute_ensemble_prediction(self, tree_predictions: np.ndarray) -> np.ndarray:
        return np.mean(tree_predictions, axis=1)
    
class MyRandomForestClassifier(RandomForestEstimator):
    def _init_decision_tree(self):
        assert self.criterion is not None, "Необходимо задать меру чистоты"
        return DecisionTreeClassifier(max_depth = self.max_depth,
                                      min_samples_split = self.min_samples_split,
                                      max_features = self.max_features,
                                      criterion = self.criterion)

    def _compute_oob_score(self, y_true: np.ndarray, oob_predictions: DefaultDict) -> float:
        true_labels = np.array([y_true[idx] for idx in oob_predictions.keys()])
        oob_prediction = np.array([np.bincount(predictions).argmax() for predictions in oob_predictions.values()])
        return accuracy_score(true_labels, oob_prediction)

    def _compute_ensemble_prediction(self, tree_predictions: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda pred: np.bincount(pred).argmax(), axis=1, arr=tree_predictions.astype(int))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        '''
        :param X: матрица независимых переменных
        :return: вектор предсказаний
        '''
        tree_predictions = []
        for decision_tree in self._decision_trees:
            prediction = decision_tree.predict(X).reshape(-1,1)
            tree_predictions.append(prediction)
        tree_predictions = np.concatenate(tree_predictions, axis=1)
        return np.mean(tree_predictions, axis=1)