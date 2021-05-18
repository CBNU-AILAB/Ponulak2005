import argparse

import numpy as np

from util import DataLoader, LossReSuMe, OptimizerReSuMe, rasterplot
from encoder import PoissonEncoder, LabelEncoder
from network import SNN


def app(opt):
    train_loader = DataLoader(data=opt.data, batch_size=opt.batch_size, shuffle=True, train=True)

    encoder_img = PoissonEncoder()
    encoder_lb = LabelEncoder(opt.num_classes, tau_ref=0.002)

    model = SNN()

    loss = LossReSuMe(W=model.pop2.W)

    optimizer = OptimizerReSuMe(W=model.pop2.W, delta_W=loss.delta_W, lr=0.01)

    num_steps = int(opt.simulation_times/opt.time_steps)

    for step in range(1, num_steps):
        t = step * opt.time_steps

        # image.shape: [1, 1, 28, 28]
        # label.shape: [1]
        # TODO: step이 1부터 시작해서 첫 데이터는 한 번 부족함. 이런 것들 보완해야함 (시작과 마지막 부분)
        if step % int(opt.time_frames/opt.time_steps) == 0:
            train_loader.next()
        image, label = train_loader()
        image = image.numpy()
        label = label.numpy()
        image = np.squeeze(image, 0)  # image.shape: [1, 28, 28]

        spiked_image = encoder_img(image, trace=True)
        spiked_label = encoder_lb(t, label, trace=True)

        model(t, spiked_image)

        loss(t=t, S_in=model.pop1.s, S_l=model.pop2.s, S_d=spiked_label)

        optimizer.step()

    a = np.array(encoder_lb.records)
    d = np.array(spiked_label)
    s1 = np.array(model.pop1.s)
    s2 = np.array(model.pop2.s)

    time = np.arange(0, num_steps)

    print(time.size)
    print(s2.size)

    rasterplot(time, s2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--simulation_times', default=2, type=float)  # 2s
    parser.add_argument('--time_steps', default=0.001, type=float)  # 1ms
    parser.add_argument('--time_frames', default=0.05, type=float)  # 50ms

    app(parser.parse_args())
