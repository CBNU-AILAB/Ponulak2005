import numpy as np


class Encoder:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class PoissonEncoder(Encoder):
    def __init__(self):
        super(PoissonEncoder, self).__init__()
        self.records = []

    def __call__(self, image, trace=False):
        # image.size: [c, h, w]
        # spike_image.size: [c, h, w]
        c, h, w = image.shape
        spike_image = np.random.uniform(size=(c, h, w))
        for i in range(c):
            for j in range(h):
                for k in range(w):
                    if spike_image[i, j, k] < image[i, j, k]:
                        spike_image[i, j, k] = 1
                    else:
                        spike_image[i, j, k] = 0
        self.records.append(spike_image)
        if trace:
            return self.records
        return spike_image


class LabelEncoder(Encoder):
    def __init__(self, num_classes, tau_ref):
        super(LabelEncoder, self).__init__()
        self.num_classes = num_classes
        self.tau_ref = tau_ref
        self.records = [np.zeros(self.num_classes)]

    def __call__(self, t, label, trace=False):
        o = np.zeros(self.num_classes)
        tt = int(t*1000)
        rr = int(self.tau_ref*1000+1)
        if tt%rr == 0:
            o[label] = 1
        self.records.append(o)
        if trace:
            return self.records
        return o
