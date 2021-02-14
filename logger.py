from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)

    def log_image(self, name, image, step):
        self.writer.add_image(name, image, step)
