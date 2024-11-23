import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable


logger = logging.getLogger('model training')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S %p')

class LinearEstimator(ABC):
    '''
    Интерфейс для линейной и логистической регрессии с оптимизацией через градиентный спуск. 
    Функционал ошибки для регрессии - MSE, классификации - LogLoss.
    '''
    def __init__(self, n_iter: int = 1000, learning_rate: float | Callable = 0.1, tolerance: float = 1e-4,
                 penalty: str | None = None, alpha: float = 1.0, verbose: bool = False):
        '''
        :param n_iter: кол-во итераций градиентного спуска
        :param learning_rate: скорость обучения градиентного спуска
        :param tolerance: порог для критерия останова
        :param penalty: тип регуляризации. Возможные значения None(OLS), "L1"(Lasso), "L2"(Ridge)
        :param alpha: коэффициент регуляризации
        :param verbose: вывод промежуточной информации об обучении на каждой 10-ой итерации градиентного спуска
        '''
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        assert penalty in [None, "L1", "L2"], "Некорректный ввод. Допустимые значения: None, L1, L2"
        self.penalty = penalty
        self.alpha = alpha
        self.verbose = verbose

    @staticmethod
    def add_constant(X: np.ndarray) -> np.ndarray:
        '''
        :return: исходный набор данных с добавленным полем-константой под intercept
        '''
        return np.hstack((np.ones(X.shape[0]).reshape(-1,1), X))

    @abstractmethod
    def _compute_loss(self, weights: np.ndarray) -> float:
        '''
        :param weights: вектор весов
        :return: функционал ошибки с учетом заданных параметров
        '''
        pass

    @abstractmethod
    def _compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        '''
        :param weights: вектор весов
        :return: градиент функционала ошибки с учетом заданных параметров
        '''
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        :param X: матрица независимых переменных
        :param y: зависимая переменная
        '''
        self._X, self._y = self.add_constant(X).copy(), y.copy()
        self.n, self.p = self._X.shape
        self.fitted_weights = self._start_gradient_descent()
        
    def _start_gradient_descent(self) -> np.ndarray:
        '''
        Запускает алгоритм градиентного спуска 

        :return: подобранный вектор весов
        '''
        weights = np.random.normal(size=self.p, scale=1)
        self.objective_path = [self._compute_loss(weights)] # функционал ошибки на всем пути расчета градиента
        self.weights_path = [weights] # веса на всем пути расчета градиента
        for epoch in range(self.n_iter):
            if callable(self.learning_rate): # если хотим задать динамическую скорость обучения
                step = self.learning_rate(epoch) * self._compute_gradient(weights)
            else:
                step = self.learning_rate * self._compute_gradient(weights)
            weights = weights - step
            if np.linalg.norm(weights - self.weights_path[-1]) < self.tolerance: # критерий останова, проверяем, что w(t) больше w(t-1) на эпсилон
                logger.info(f'Алгоритм сошелся. Кол-во итераций: {epoch}')
                break
            self.weights_path.append(weights)
            if self.penalty == "L1":
                self.weights_path[-1][self.weights_path[-1] < self.tolerance] = 0 # если вес меньше эпсилон, округляем его до нуля
            self.objective_path.append(self._compute_loss(self.weights_path[-1]))
            if self.verbose == True and epoch % 10 == 0:
                logger.info(f'Итерация: {epoch}; Функционал ошибки: {self._compute_loss(self.weights_path[-1])}')
        return self.weights_path[-1]

    @abstractmethod
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        '''
        :param X: матрица независимых переменных
        :param threshold: порог для конвертации вероятностей в лейблы в задаче классификации
        :return: вектор предсказаний
        '''
        pass

class MyLinearRegression(LinearEstimator):
    def _compute_loss(self, weights: np.ndarray) -> float:
        if self.penalty is None:
            return np.mean(np.square(np.dot(self._X, weights) - self._y))
        elif self.penalty == "L1":
            return np.mean(np.square(np.dot(self._X, weights) - self._y)) + self.alpha * np.sum(np.abs(weights[1:])) # weights[1:] т.к. не учитываем intercept
        elif self.penalty == "L2":
            alpha_scaled = self.alpha / self.n # так реализовано в sklearn, solver:sag
            return np.mean(np.square(np.dot(self._X, weights) - self._y)) + alpha_scaled * np.sum(np.square(weights[1:]))

    def _compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        if self.penalty is None:
            return np.dot(self._X.T, (np.dot(self._X, weights) - self._y)) / self.n
        elif self.penalty == "L1":
            return np.dot(self._X.T, (np.dot(self._X, weights) - self._y)) / self.n + np.hstack([0, self.alpha * np.sign(weights[1:])])
        elif self.penalty == "L2":
            alpha_scaled = self.alpha / self.n
            return np.dot(self._X.T, (np.dot(self._X, weights) - self._y)) / self.n + np.hstack([0, alpha_scaled * weights[1:]])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        _X = self.add_constant(X)
        return np.dot(_X, self.fitted_weights)
    
class MyLogisticRegression(LinearEstimator):
    @staticmethod
    def sigmoid(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        '''
        :param X: матрица независимых переменных
        :param weights: вектор весов
        :return: вектор вероятностей принадлежности к 1 классу
        '''
        return 1 / (1 + np.exp(-np.dot(X, weights)))

    def _compute_loss(self, weights: np.ndarray) -> float:
        if self.penalty is None:
            ones = self._y * np.log(self.sigmoid(self._X, weights) + 1e-9)
            zeros = (1 - self._y) * np.log(1 - self.sigmoid(self._X, weights) + 1e-9)
            return -np.mean(ones + zeros)
        if self.penalty == "L1":
            pass # не стал реализовывать, т.к. тоже самое, что и в линейной регрессии
        if self.penalty == "L2":
            pass

    def _compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        if self.penalty is None:
            return np.dot(self._X.T, (self.sigmoid(self._X, weights) - self._y)) / self.n
        if self.penalty == "L1":
            pass
        if self.penalty == "L2":
            pass

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        X = self.add_constant(X)
        probabilities = self.sigmoid(X, self.fitted_weights)
        return np.array([1 if p > threshold else 0 for p in probabilities])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        '''
        :param X: матрица независимых переменных
        :return: вектор вероятностей принадлежности к 1 классу
        '''
        X = self.add_constant(X)
        probabilities = self.sigmoid(X, self.fitted_weights)
        return probabilities