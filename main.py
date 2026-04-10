from types import SimpleNamespace
from train import train
from test import test


def main():
    config = SimpleNamespace(
        mode="train",
        data_dir="data/raw",
        save_dir="saved_models",
        batch_size=32,
        epochs=5,
        lr=1e-3,
        limit=None,
        num_workers=2,
        val_every=2,
    )

    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        raise ValueError("mode must be either 'train' or 'test'")

if __name__ == "__main__":
    main()
