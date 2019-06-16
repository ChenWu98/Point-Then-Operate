# import tensorflow as tf
from tensorboardX import SummaryWriter

from config import Config
config = Config()


class Logger(object):
    """Using tensorboardX such that need no dependency on tensorflow."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def upload_image(self, tag, image, step):
        """image: (3, H, W)"""
        self.writer.add_image(tag, image, step)

    def upload_text(self, tag, text, step):
        self.writer.add_text(tag, text, step)

    def upload_figure(self, tag, fig, step):
        self.writer.add_figure(tag, fig, step)
