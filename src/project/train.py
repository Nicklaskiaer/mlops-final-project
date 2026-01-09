from project.model import Model
from project.data import MyDataset


def train():
    dataset = MyDataset("data/raw")  # noqa: F841
    model = Model()  # noqa: F841
    # add rest of your training code here


if __name__ == "__main__":
    train()
