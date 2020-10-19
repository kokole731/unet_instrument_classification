from torch.utils.tensorboard import SummaryWriter

class Report():
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.close()

    def advance_step(self):
        self.step += 1

    def add_scalar(self, tag, obj):
        self.writer.add_scalar(tag, obj, self.step)