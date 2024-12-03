import logging
from typing import DefaultDict, Union, Optional, List
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import r2_score, accuracy_score

logger = logging.getLogger('model training')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S %p')

class RandomForestEstimator(ABC):
    '''
    Интерфейс для случайного леса для регрессии и классификации с OOB оценкой
    '''

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 min_samples_split: int = 2, max_features: str = "sqrt", criterion: Optional[str] = None):
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
    def _init_decision_tree(self) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
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
        for _ in range(self.n_estimators):
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
        pred = []
        for decision_tree in self._decision_trees:
            tree_pred = decision_tree.predict(X).reshape(-1,1)
            pred.append(tree_pred)
        pred = np.concatenate(pred, axis=1)
        return self._compute_ensemble_prediction(pred)

    @abstractmethod
    def _compute_ensemble_prediction(self, predictions: np.ndarray) -> np.ndarray:
        '''
        :return среднее или мажоритарный класс по n_estimators предсказаниям
        '''
        pass

class MyRandomForestRegressor(RandomForestEstimator):
    def _init_decision_tree(self) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
        assert self.criterion is not None, "Необходимо задать меру чистоты"
        return DecisionTreeRegressor(max_depth = self.max_depth,
                                     min_samples_split = self.min_samples_split,
                                     max_features = self.max_features,
                                     criterion = self.criterion)

    def _compute_oob_score(self, y_true: np.ndarray, oob_predictions: DefaultDict) -> float:
        true_labels = np.array([y_true[idx] for idx in oob_predictions.keys()])
        oob_prediction = np.array([np.mean(predictions) for predictions in oob_predictions.values()])
        return r2_score(true_labels, oob_prediction)

    def _compute_ensemble_prediction(self, predictions: np.ndarray) -> np.ndarray:
        return np.mean(predictions, axis=1)
    
class MyRandomForestClassifier(RandomForestEstimator):
    def _init_decision_tree(self) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
        assert self.criterion is not None, "Необходимо задать меру чистоты"
        return DecisionTreeClassifier(max_depth = self.max_depth,
                                      min_samples_split = self.min_samples_split,
                                      max_features = self.max_features,
                                      criterion = self.criterion)

    def _compute_oob_score(self, y_true: np.ndarray, oob_predictions: DefaultDict) -> float:
        true_labels = np.array([y_true[idx] for idx in oob_predictions.keys()])
        oob_prediction = np.array([np.bincount(predictions).argmax() for predictions in oob_predictions.values()])
        return accuracy_score(true_labels, oob_prediction)

    def _compute_ensemble_prediction(self, predictions: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda pred: np.bincount(pred).argmax(), axis=1, arr=predictions.astype(int))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        '''
        :param X: матрица независимых переменных
        :return: вектор предсказаний
        '''
        pred = []
        for decision_tree in self._decision_trees:
            tree_pred = decision_tree.predict(X).reshape(-1,1)
            pred.append(tree_pred)
        pred = np.concatenate(pred, axis=1)
        return np.mean(pred, axis=1)

class GradientBoostingEstimator(ABC):
    '''
    Интерфейс для градиентного бустинга для классификации и регрессии.
    Функционал ошибки для регрессии - MSE, для классификации - LogLoss.
    '''

    def __init__(self, learning_rate: float = 0.1, n_estimators: int = 100, 
                 min_samples_split: int = 2, max_depth: int = 3, criterion: Optional[str] = None):
        '''
        :param learning_rate: скорость обучения
        :param n_estimators: кол-во моделей в ансамбле
        :param min_samples_split: минимальное кол-во наблюдений в ноде для сплита
        :param max_depth: максимальная глубина дерева
        '''
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self._decision_trees = []

    @abstractmethod
    def _init_decision_tree(self) -> DecisionTreeRegressor:
        '''
        :return: дерево решений
        '''
        pass

    @abstractmethod
    def _compute_initial_pred(self, y: np.ndarray) -> float:
        '''
        :return: первичное предсказание в ансамбле
        '''
        pass

    @abstractmethod
    def _update_leaf_nodes(self, 
                           tree: DecisionTreeRegressor, 
                           X: np.ndarray, y: np.ndarray, 
                           residuals: np.ndarray):
        '''
        Расчитывает оптимальное значение gamma для каждой листовой ноды (формула №18 в статье Фридмана)

        :param tree: дерево решений
        :param X: матрица независимых переменных
        :param y: зависимая переменная
        :param residuals: антиградиент функции потерь 
        '''
        pass

    @abstractmethod
    def _transform_ensemble_pred(self, ensemble_pred: np.ndarray) -> np.ndarray:
        '''
        Преобразовывает предсказание ансамбля.

        :param ensemble: предсказание ансамбля
        :return: преобразованное предсказание ансамбля
        '''
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        :param X: матрица зависимых переменных
        :param y: независимая переменная
        '''
        self.initial_pred = self._compute_initial_pred(y)
        ensemble_pred = np.full(X.shape[0], self.initial_pred)
        for _ in range(self.n_estimators):
            residuals = y - self._transform_ensemble_pred(ensemble_pred)
            decision_tree = self._init_decision_tree()
            decision_tree.fit(X, residuals)
            self._update_leaf_nodes(decision_tree.tree_, X, y, residuals)
            self._decision_trees.append(decision_tree)
            ensemble_pred += decision_tree.predict(X)

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        :param X: матрица независимых перменных
        :return: вектор предсказаний
        '''
        pass

class MyGradientBoostingRegressor(GradientBoostingEstimator):
    def _init_decision_tree(self)  -> DecisionTreeRegressor:
        assert self.criterion is not None, "Необходимо задать меру чистоты"
        return DecisionTreeRegressor(
                    max_depth = self.max_depth,
                    min_samples_split = self.min_samples_split,
                    criterion = self.criterion)

    def _compute_initial_pred(self, y: np.ndarray) -> float:
        return np.mean(y)

    def _update_leaf_nodes(self, 
                           tree: DecisionTreeRegressor, 
                           X: np.ndarray, y: np.ndarray, 
                           residuals: np.ndarray):
        X = X.astype(np.float32)
        leafs = tree.apply(X)
        for leaf in np.unique(leafs):
            tree.value[leaf, 0, 0] *= self.learning_rate          

    def _transform_ensemble_pred(self, ensemble_pred: np.ndarray) -> np.ndarray:
        return ensemble_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        ensemble_pred = np.full(X.shape[0], self.initial_pred)
        for i in range(self.n_estimators):
            ensemble_pred += self._decision_trees[i].predict(X)
        return ensemble_pred

class MyGradientBoostingClassifier(GradientBoostingEstimator):
    def _init_decision_tree(self)  -> DecisionTreeRegressor:
        assert self.criterion is not None, "Необходимо задать меру чистоты"
        return DecisionTreeRegressor(
                    max_depth = self.max_depth,
                    min_samples_split = self.min_samples_split,
                    criterion = self.criterion)

    def _compute_initial_pred(self, y: np.ndarray) -> float:
        return 0. # 0. log odds == 0.5 proba

    def _update_leaf_nodes(self, 
                           tree: DecisionTreeRegressor, 
                           X: np.ndarray, y: np.ndarray, 
                           residuals: np.ndarray):
        X = X.astype(np.float32)
        leafs = tree.apply(X)
        for leaf in np.unique(leafs):
            idx = np.nonzero(leafs == leaf)[0]
            p = y[idx] - residuals[idx]
            assert not np.any((p < 0) | (p > 1)), "вероятность вне допустимого диапазона [0,1] при расчете значения листа"
            numerator = np.mean(residuals[idx])
            denominator = np.mean(p * (1 - p))
            gamma = numerator / denominator
            tree.value[leaf, 0, 0] = self.learning_rate * gamma

    def _transform_ensemble_pred(self, ensemble_pred: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-ensemble_pred))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ensemble_pred = np.full(X.shape[0], self.initial_pred)
        for i in range(self.n_estimators):
            ensemble_pred += self._decision_trees[i].predict(X)
        return self._transform_ensemble_pred(ensemble_pred)

    def predict(self, X: np.ndarray) -> np.ndarray:
        ensemble_pred = np.full(X.shape[0], self.initial_pred)
        for i in range(self.n_estimators):
            ensemble_pred += self._decision_trees[i].predict(X)
        return (ensemble_pred > 0.).astype(int)