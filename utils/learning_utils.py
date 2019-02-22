
class MyConfig:
    def __init__(self, n_past=8, n_next=8):
        self.n_past = n_past
        self.n_next = n_next


def adjust_learning_rate(optimizer, epoch):
    lr = 0.005

    # if epoch > 1000:
    #     lr = lr / 100000
    # elif epoch > 800:
    #     lr = lr / 10000
    # elif epoch > 400:
    #     lr = lr / 1000
    # elif epoch > 200:
    #     lr = lr / 200
    # elif epoch > 100:
    #     lr = lr / 20
    # elif epoch > 40:
    #     lr = lr / 5

    lr *= (0.6 ** (epoch // 50))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
