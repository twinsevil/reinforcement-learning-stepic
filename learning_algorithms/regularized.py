from learning_algorithms.network import Network, sigmoid, sigmoid_prime
import numpy as np

class RegularizedNetwork(Network):
    def __init__(self, sizes, output_function=sigmoid, output_derivative=sigmoid_prime, l1=0, l2=0):
        super().__init__(sizes, output_function, output_derivative)
        self.l1 = l1
        self.l2 = l2

    def update_mini_batch(self, mini_batch, eta):
        """
        Обновить веса и смещения нейронной сети, сделав шаг градиентного
        спуска на основе алгоритма обратного распространения ошибки, примененного
        к одному mini batch. Учесть штрафы за L1 и L2.
        ``mini_batch`` - список кортежей вида ``(x, y)``,
        ``eta`` - величина шага (learning rate).
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        eps = eta / len(mini_batch)
        self.weights = [w - eps * nw - self.l1 * np.sign(w) - self.l2 * w for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eps * nb for b, nb in zip(self.biases, nabla_b)]


    def cost_function2(self, test_data, l1, l2):
        c = 0
        for example, y in test_data:
            yhat = self.feedforward(example)
            c += np.sum((y - yhat) ** 2)
            w1 = 0
            w2 = 0
            for arr in self.weights:
                for line in arr:
                    for el in line:
                        w1 += np.abs(el)
                        w2 += el ** 2
        n = len(test_data)
        return c / n + l1 * w1 + l2 * w2