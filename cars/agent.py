import random
from abc import ABCMeta, abstractmethod
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from cars.utils import Action
from learning_algorithms.network import Network
from learning_algorithms.regularized import RegularizedNetwork

class Agent(metaclass=ABCMeta):
    @property
    @abstractmethod
    def rays(self):
        pass

    @abstractmethod
    def choose_action(self, sensor_info):
        pass

    @abstractmethod
    def receive_feedback(self, reward):
        pass


class SimpleCarAgent(Agent):

    filename = ""

    def __init__(self, history_data=int(5000)):
        """
        Создаёт машинку
        :param history_data: количество хранимых нами данных о результатах предыдущих шагов
        """
        self.evaluate_mode = False  # этот агент учится или экзаменутеся? если учится, то False
        self._rays = 7  # выберите число лучей ладара; например, 5
        # here +2 is for 2 inputs from elements of Action that we are trying to predict
        self.neural_net = RegularizedNetwork([self.rays + 4, 50, 20, 10,
                                   # внутренние слои сети: выберите, сколько и в каком соотношении вам нужно
                                   # например, (self.rays + 4) * 2 или просто число
                                   1],
                                   output_function=lambda x: x, output_derivative=lambda x: 1, l1=0.0005, l2=0.00)
        self.sensor_data_history = deque([], maxlen=history_data)
        self.chosen_actions_history = deque([], maxlen=history_data)
        self.reward_history = deque([], maxlen=history_data)
        self.step = 0
        self.cost_train = 0
        self.cost_test = 0


    @classmethod
    def from_weights(cls, layers, weights, biases):
        """
        Создание агента по параметрам его нейронной сети. Разбираться не обязательно.
        """
        agent = SimpleCarAgent()
        agent._rays = weights[0].shape[1] - 4
        nn = RegularizedNetwork(layers, output_function=lambda x: x, output_derivative=lambda x: 1,  l1=0.000, l2=0)

        if len(weights) != len(nn.weights):
            raise AssertionError("You provided %d weight matrices instead of %d" % (len(weights), len(nn.weights)))
        for i, (w, right_w) in enumerate(zip(weights, nn.weights)):
            if w.shape != right_w.shape:
                raise AssertionError("weights[%d].shape = %s instead of %s" % (i, w.shape, right_w.shape))
        nn.weights = weights

        if len(biases) != len(nn.biases):
            raise AssertionError("You provided %d bias vectors instead of %d" % (len(weights), len(nn.weights)))
        for i, (b, right_b) in enumerate(zip(biases, nn.biases)):
            if b.shape != right_b.shape:
                raise AssertionError("biases[%d].shape = %s instead of %s" % (i, b.shape, right_b.shape))
        nn.biases = biases

        agent.neural_net = nn

        return agent

    @classmethod
    def from_string(cls, s):
        from numpy import array  # это важный импорт, без него не пройдёт нормально eval
        layers, weights, biases = eval(s.replace("\n", ""), locals())
        return cls.from_weights(layers, weights, biases)

    @classmethod
    def from_file(cls, filename):
        cls.filename = filename
        c = open(filename, "r").read()
        return cls.from_string(c)

    def show_weights(self):
        params = self.neural_net.sizes, self.neural_net.weights, self.neural_net.biases
        #np.set_printoptions(threshold=np.nan)
        return repr(params)

    def to_file(self, filename):
        c = self.show_weights()
        f = open(filename, "w")
        f.write(c)
        f.close()

    @property
    def rays(self):
        return self._rays

    def choose_action(self, sensor_info):
        # хотим предсказать награду за все действия, доступные из текущего состояния
        rewards_to_controls_map = {}
        # дискретизируем множество значений, так как все возможные мы точно предсказать не сможем
        for steering in np.linspace(-1, 1, 3):  # выбирать можно и другую частоту дискретизации, но
            for acceleration in np.linspace(-0.75, 0.75, 3):  # в наших тестах будет именно такая
                action = Action(steering, acceleration)
                agent_vector_representation = np.append(sensor_info, action)
                agent_vector_representation = agent_vector_representation.flatten()[:, np.newaxis]
                predicted_reward = float(self.neural_net.feedforward(agent_vector_representation))
                rewards_to_controls_map[predicted_reward] = action

        # ищем действие, которое обещает максимальную награду
        rewards = list(rewards_to_controls_map.keys())
        highest_reward = max(rewards)
        best_action = rewards_to_controls_map[highest_reward]

        # Добавим случайности, дух авантюризма. Иногда выбираем совершенно
        # рандомное действие
        if (not self.evaluate_mode) and (random.random() < 0.05):
            highest_reward = rewards[np.random.choice(len(rewards))]
            best_action = rewards_to_controls_map[highest_reward]
        # следующие строки помогут вам понять, что предсказывает наша сеть
        #    print("Chosen random action w/reward: {}".format(highest_reward))
        # else:
        #    print("Chosen action w/reward: {}".format(highest_reward))

        # запомним всё, что только можно: мы хотим учиться на своих ошибках
        self.sensor_data_history.append(sensor_info)
        self.chosen_actions_history.append(best_action)
        self.reward_history.append(0.0)  # мы пока не знаем, какая будет награда, это
        # откроется при вызове метода receive_feedback внешним миром
        #best_action = Action(-1, 0.75)
        return best_action

    def receive_feedback(self, reward, train_every=50, reward_depth=7):
        """
        Получить реакцию на последнее решение, принятое сетью, и проанализировать его
        :param reward: оценка внешним миром наших действий
        :param train_every: сколько нужно собрать наблюдений, прежде чем запустить обучение на несколько эпох
        :param reward_depth: на какую глубину по времени распространяется полученная награда
        """
        # считаем время жизни сети; помогает отмерять интервалы обучения
        self.step += 1

        # начиная с полной полученной истинной награды,
        # размажем её по предыдущим наблюдениям
        # чем дальше каждый раз домножая её на 1/2
        # (если мы врезались в стену - разумно наказывать не только последнее
        # действие, но и предшествующие)
        i = -1
        while len(self.reward_history) > abs(i) and abs(i) < reward_depth:
            self.reward_history[i] += reward
            reward *= 0.5
            i -= 1



        # Если у нас накопилось хоть чуть-чуть данных, давайте потренируем нейросеть
        # прежде чем собирать новые данные
        # (проверьте, что вы в принципе храните достаточно данных (параметр `history_data` в `__init__`),
        # чтобы условие len(self.reward_history) >= train_every выполнялось
        if not self.evaluate_mode and (len(self.reward_history) >= train_every) and not (self.step % train_every):
            train_index = np.random.choice([True, False], len(self.sensor_data_history), replace=True, p=[0.99, 0.01])
            #X_train = np.concatenate([self.sensor_data_history, self.chosen_actions_history], axis=1)
            #y_train = self.reward_history
            X = np.concatenate([self.sensor_data_history, self.chosen_actions_history], axis=1)
            y = self.reward_history

            #X_train = X[np.logical_not(test_index)]
            #y_train = y[np.logical_not(test_index)]
            train_data = np.array([(x[:, np.newaxis], y) for x, y in zip(X, y)])[train_index]
            test_data = np.array([(x[:, np.newaxis], y) for x, y in zip(X, y)])[np.logical_not(train_index)]

            mini_batch_size = train_every if len(train_data) > train_every else len(train_data)
            self.neural_net.SGD(training_data=train_data.copy(), epochs=20, mini_batch_size=train_every, eta=0.01)

            """
            Визуализация процесса обучения (функции потерь)
            """
        
            cost_train = []
            cost_test = []
            cost_train.append(self.neural_net.cost_function2(train_data, self.neural_net.l1, self.neural_net.l2))
            #cost_test.append(self.neural_net.cost_function2(test_data, self.neural_net.l1, self.neural_net.l2))
            self.cost_train = cost_train[-1]
            #self.cost_test = cost_test[-1]
            '''
            predictions = []
            for example, y in train_data:
                predictions.append(self.neural_net.feedforward(example))
            '''
            if self.step % 1500 == 0:
                for _ in range(20):
                    self.neural_net.SGD(training_data=train_data.copy(), epochs=1, mini_batch_size=mini_batch_size, eta=0.05)
                    cost_train.append(
                        self.neural_net.cost_function2(train_data, self.neural_net.l1, self.neural_net.l2))
                    cost_test.append(self.neural_net.cost_function2(test_data, self.neural_net.l1, self.neural_net.l2))
                fig = plt.figure(figsize=(10, 5))
                fig.add_subplot(1, 1, 1)
                plt.plot(cost_train, label="Training error", color="orange")
                plt.plot(cost_test, label="Test error", color="blue")
                plt.title("Learning curve")
                plt.ylabel("Cost function")
                plt.xlabel("Epoch number")
                plt.legend()
                plt.show()


