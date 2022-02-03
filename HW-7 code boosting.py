from sklearn.base import BaseEstimator
from collections import defaultdict

import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

sns.set(style='darkgrid')


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting(BaseEstimator):

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor  # базовые модели --- деревья
        self.base_model_params: dict = {} if base_model_params is None else base_model_params  # гиперпараметры

        self.n_estimators: int = n_estimators  # число обучаемых моделей

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate  # темп обучения
        self.subsample: float = subsample  # число объектов в бутстрапированной выборке

        self.early_stopping_rounds: int = early_stopping_rounds  # число итераций, после к-х перестаем обучаться, если в
        # течение них лосс не перестал не улучшаться

        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot  # рисовать ли график

        self.history = defaultdict(list)
        # self.history = {'train': [], 'valid': []}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.loss_derivative2 = lambda y, z: y ** 2 * self.sigmoid(-y * z) * (1 - self.sigmoid(-y * z))

    def fit_new_base_model(self, x, y, predictions):
        # создаем бутстрапированную выборку
        len_boot = int(self.subsample * x.shape[0])  # размер бутстрапированной выборки
        ind_boot = np.random.choice(x.shape[0], size=len_boot)  # выбранные индексы
        x_boot = x[ind_boot]  # бутстрапированная выборка
        y_boot = y[ind_boot]  # бустрапированная выборка

        # обучаем модель на бутстрапированной выборке
        model = self.base_model_class(**self.base_model_params)
        model.fit(x_boot, y_boot)

        # ищем гамму
        gamma = self.find_optimal_gamma(y, old_predictions=predictions, new_predictions=model.predict(x))

        self.gammas = np.append(self.gammas, self.learning_rate * gamma)
        self.models = np.append(self.models, model)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        """
        :param x_train: features array (train set) --- тут обучаем
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set) --- считаем качество для ранней остановки
        :param y_valid: targets array (validation set)
        """

        if x_valid is None:
            x_valid = x_train.copy()
        if y_valid is None:
            y_valid = y_train.copy()

        # нулевая модель выдает все нули (обучаем выдавать)
        model = self.base_model_class()
        model.fit(x_train, np.zeros(x_train.shape[0]))
        self.gammas = np.append(self.gammas, 1)  # берем нулевую модель с коэффициентом 1
        self.models = np.append(self.models, model)

        train_predictions = model.predict(x_train)
        valid_predictions = model.predict(x_valid)

        # тут лоссы будем хранить
        train_losses = np.array([self.loss_fn(y_train, train_predictions)])
        valid_losses = np.array([self.loss_fn(y_valid, valid_predictions)])

        loss_up = 0  # сколько раз подряд лосс увеличился на валидации

        for i in range(1, self.n_estimators):  # 1 модель уже построили --- нулевую
            # s = -dL(y, z)/dz|_{z = a_{N - 1}(x_i)}
            s = -self.loss_derivative(y=y_train, z=train_predictions)
            # обучаем новую модель, она хранится в self.models
            self.fit_new_base_model(x=x_train, y=s, predictions=train_predictions)
            model = self.models[-1]

            # обновляем предсказания на обучении и валидации
            predictions_train_one = model.predict(x_train)
            train_predictions += predictions_train_one * self.gammas[-1]
            predictions_valid_one = model.predict(x_valid)
            valid_predictions += predictions_valid_one * self.gammas[-1]

            # считаем ошибку на обучающей выборке
            loss_train = self.loss_fn(y_train, train_predictions)
            train_losses = np.append(train_losses, loss_train)
            # на валидационной
            loss_val = self.loss_fn(y_valid, valid_predictions)
            valid_losses = np.append(valid_losses, loss_val)

            # записываем ROC-AUC на тесте и валидации
            self.history['train'].append(self.score(x_train, y_train))
            self.history['valid'].append(self.score(x_valid, y_valid))

            # проверяем на раннюю остановку
            if self.early_stopping_rounds is not None:
                if valid_losses[-1] >= valid_losses[-2]:
                    loss_up += 1
                else:
                    loss_up = 0  # нам self.early_stopping_rounds _подряд_ надо, чтобы было плохо, чтобы остановиться

                if loss_up == self.early_stopping_rounds:
                    self.n_estimators = i  # успели только столько моделей обучить
                    break

        if self.plot:
            # нарисуем график ROC-AUC на трейне и валидации
            plt.plot([0] + self.history['train'], label='ROC-AUC на трейне')
            plt.plot([0] + self.history['valid'], label='ROC-AUC на валидации')
            plt.title("ROC-AUC")
            plt.xlabel("Число базовых моделей")
            plt.ylabel("ROC-AUC")

            plt.legend()
            plt.show()

            # или можно было loss
            # plt.plot(train_losses, label='loss на трейне)
            # plt.plot(valid_losses, label='loss на валидации')
            # plt.title('loss на трейне и валидации')
            # plt.xlabel('Число базовых моделей')
            # plt.ylabel('loss')

            # plt.legend()
            # plt.show()

    def predict_proba(self, x):
        predictions = np.zeros(x.shape[0])  # тут будут предсказания
        for gamma, model in zip(self.gammas, self.models):
            prediction = model.predict(x)
            predictions += prediction * gamma

        result = np.ones((x.shape[0], 2))
        result[:, 1] = self.sigmoid(predictions)  # возвращаем вероятности
        result[:, 0] -= result[:, 1]

        return result

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        # считается по сумме с коэффициентами всех построенных моделей
        return score(self, x, y)

    @property
    def feature_importances_(self):
        feature_importances = np.mean([tree.feature_importances_ for tree in self.models], axis=0)
        feature_importances /= sum(feature_importances)

        return feature_importances

