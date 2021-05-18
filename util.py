import numpy as np

import matplotlib.pyplot as plt

import torch
import torchvision


class OptimizerReSuMe:
    def __init__(self, W, delta_W, lr=0.1):
        self.W = W
        self.delta_W = delta_W
        self.lr = lr

    def step(self):
        self.W[:] = self.W + self.delta_W*self.lr


class LossReSuMe:
    def __init__(self, W, a_d=0, a_l=0, A_d=1, A_l=1, W_d=None, W_l=None, tau_d=1, tau_l=1):
        self.W = W
        self.delta_W = np.zeros(W.shape)
        self.a_d = a_d
        self.a_l = a_l
        self.A_d = A_d
        self.A_l = A_l
        self.tau_d = tau_d
        self.tau_l = tau_l
        if W_d is None:
            self.W_d = self.W_d_f
        else:
            self.W_d = W_d
        if W_l is None:
            self.W_l = self.W_l_f
        else:
            self.W_l = W_l

    def W_d_f(self, s_d):
        if s_d > 0:
            return self.A_d * np.exp(-s_d/self.tau_d)
        return 0

    def W_l_f(self, s_l):
        if s_l > 0:
            return -self.A_l * np.exp(-s_l/self.tau_l)
        return 0

    def __call__(self, t, S_in, S_l, S_d):
        for k in range(S_in[-1].shape[0]):
            for i in range(S_l[-1].shape[0]):
                delta_w = 0
                if S_l[-1][i] == 1:
                    delta_w_l = self.a_l
                    for j in range(len(S_in)):
                        s_l = t-j*0.001
                        delta_w_l += self.W_l(s_l) * S_in[j][k]
                    delta_w += delta_w_l
                if S_d[-1][i] == 1:
                    delta_w_d = self.a_d
                    for j in range(len(S_in)):
                        s_d = t-j*0.001
                        delta_w_d += self.W_d(s_d) * S_in[j][k]
                    delta_w += delta_w_d
                # print(delta_w)
                self.delta_W[i][k] = delta_w


class DataLoader:
    def __init__(self, data, batch_size, shuffle, train):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train
        self.loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                self.data,
                train=self.train,
                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
            batch_size=self.batch_size,
            shuffle=self.shuffle)
        self.it = iter(self.loader)
        self.o = next(self.it)

    def next(self):
        self.o = next(self.it)

    def __call__(self):
        return self.o


def rasterplot(time, spikes, **kwargs):
    ax = plt.gca()

    n_spike, n_neuron = spikes.shape

    kwargs.setdefault("linestyle", "None")
    kwargs.setdefault("marker", "|")

    spiketimes = []

    for i in range(n_neuron):
        temp = time[spikes[:, i] > 0].ravel()
        spiketimes.append(temp)

    spiketimes = np.array(spiketimes)

    indexes = np.zeros(n_neuron, dtype=np.int)

    for t in range(time.shape[0]):
        for i in range(spiketimes.shape[0]):
            if spiketimes[i].shape[0] <= 0:
                continue
            if indexes[i] < spiketimes[i].shape[0] and \
                    time[t] == spiketimes[i][indexes[i]]:
                ax.plot(spiketimes[i][indexes[i]], i + 1, 'k', **kwargs)

                plt.draw()
                plt.pause(0.002)

                indexes[i] += 1


def test_rasterplot():
    total_time = 1000

    time = np.arange(0, total_time)
    spikes = np.zeros(shape=(total_time, 5))

    print(time)
    print(spikes)

    for i in range(spikes.shape[1]):
        spikes[:, i] = np.random.uniform(0, 1, total_time)
        spikes[spikes > 0.5] = 1
        spikes[spikes < 0.5] = 0

    print(time.shape)
    print(spikes.shape)

    rasterplot(time, spikes)
