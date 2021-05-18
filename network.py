import numpy as np


class PopulationLIF:
    def __init__(self, in_neurons, neurons, q=1, resting=0, threshold=1, tau_m=0.01, tau_ref=0.002):
        self.in_neurons = in_neurons
        self.neurons = neurons
        self.q = q
        self.u_r = resting
        self.th = threshold
        self.tau_m = tau_m
        self.tau_ref = tau_ref
        self.W = np.zeros((self.neurons, self.in_neurons))
        self.a = [np.zeros(self.in_neurons)]
        self.I = [np.zeros(self.neurons)]
        self.u = [np.zeros(self.neurons)]
        self.s = [np.zeros(self.neurons)]
        self.ref = np.zeros(self.neurons)
        self.t_hat = np.zeros(self.neurons)

        self.reset_parameters()

    def reset_parameters(self):
        self.W[:] = np.random.uniform(low=-0.5, high=0.5, size=(self.neurons, self.in_neurons))

    def __call__(self, t, x):
        self.psc(t=t,
                 s=x,
                 q=self.q,
                 a=self.a)
        self.current(t=t,
                     W=self.W,
                     a=self.a,
                     I=self.I)
        self.integrate(t=t,
                       I=self.I,
                       u=self.u,
                       t_hat=self.t_hat,
                       ref=self.ref,
                       u_r=self.u_r,
                       tau_m=self.tau_m)
        self.fire(t=t,
                  u=self.u,
                  t_hat=self.t_hat,
                  s=self.s,
                  ref=self.ref,
                  th=self.th,
                  u_r=self.u_r,
                  tau_ref=self.tau_ref)
        self.refractory(t=t,
                        ref=self.ref,
                        s=self.s)
        return self.s

    def psc(self, t, s, q, a):
        """
        a(t-t_) = q*d(t-t_)
        :param t: time
        :param s: presynaptic spikes
        :param q:
        :param a: postsynaptic currents (output)
        :return:
        """
        aa = np.zeros(self.in_neurons)
        aa[:] = q * s[-1]
        a.append(aa)

    def current(self, t, W, a, I):
        """
        :param t: time
        :param W: weights
        :param a: PSC
        :param I: input currents (output)
        :return:
        """
        II = np.zeros(self.neurons)
        II[:] = np.dot(W, a[-1])
        I.append(II)

    def integrate(self, t, I, u, t_hat, ref, u_r, tau_m):
        """
        :param t:
        :param I:
        :param u:
        :param t_hat:
        :param ref:
        :param u_r:
        :param tau_m:
        :return:
        """
        uu = np.zeros(self.neurons)
        for i in range(uu.shape[0]):
            if ref[i] <= 0:
                tt = int(t*1000)
                tt_hat = int(t_hat[i]*1000)
                uu[i] = u_r*np.exp(-(t-t_hat[i])/tau_m)
                for s in range(tt-tt_hat+1):
                    uu[i] += np.exp(-(s*0.001)/tau_m) * I[tt-s][i]
        u.append(uu)

    def fire(self, t, u, t_hat, s, ref, th, u_r, tau_ref):
        """
        :param t:
        :param u:
        :param t_hat:
        :param s:
        :param ref:
        :param th:
        :param u_r:
        :param tau_ref:
        :return:
        """
        ss = np.zeros(self.neurons)
        for i in range(ss.shape[0]):
            if ref[i] <= 0 and u[-1][i] > th:
                ss[i] = 1
                t_hat[i] = t
                u[-1][i] = u_r
                ref[i] += tau_ref
        s.append(ss)

    def refractory(self, t, ref, s):
        """
        :param t:
        :param ref:
        :param s:
        :return:
        """
        for i in range(ref.shape[0]):
            if s[-1][i] == 0 and ref[i] > 0:
                ref[i] -= 0.001

class SNN:
    def __init__(self):
        self.pop1 = PopulationLIF(28*28, 10)
        self.pop2 = PopulationLIF(10, 10)

    def __call__(self, t, x):
        return self._forward(t, x)

    def _forward(self, t, x):
        # x는 list일 수 있고 ndarray일 수 있다.
        # list인 경우는 trace=True인 경우로 이전 값도 모두 가져온다.
        # ndarra인 경우는 현재 값만 가져온 경우를 나타낸다.
        if isinstance(x, list):
            for i in range(len(x)):
                x[i] = x[i].reshape(-1)
        else:
            x = x.reshape(-1)
        o = self.pop1(t, x)
        o = self.pop2(t, o)
        return o