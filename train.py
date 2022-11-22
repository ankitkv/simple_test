import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import datasets, transforms


class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage=None):
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        self.dataset_train = datasets.MNIST(
            "data", transform=transform, train=True, download=True
        )

    def train_dataloader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=4,
            shuffle=shuffle,
            num_workers=1,  # changing this to 0 removes the error
            pin_memory=True,
            drop_last=True,
        )


class PretrainModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(32 * 32, 10)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())

    def training_step(self, batch, _):
        images = batch[0]

        dummy = DictConfig({"test": 1})
        # removing this, or changing "anything" to "test", removes the error
        getattr(dummy, "anything", False)

        images = images.view(images.shape[0], -1)
        out = self.model(images)
        loss = out.sum()

        self.all_gather(out)  # removing this removes the error

        return loss


def main():
    datamodule = DataModule()
    module = PretrainModule()
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_false",
        num_nodes=1,
        devices=1,
        log_every_n_steps=1,
        limit_train_batches=2,
        limit_val_batches=2,
    )
    trainer.fit(
        module,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()
