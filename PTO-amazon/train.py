from Experiment import Experiment
from config import Config

config = Config()


def train():

    # Building the wrapper
    wrapper = Experiment()

    print("------ Training from scratch. ------")
    wrapper.train()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("once")
    train()
