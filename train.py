import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import datasets, transforms


class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data_path = "data"
        self.batch_size = 64
        self.num_workers = 1

    def setup(self, stage=None):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        # with MNIST, the batch size needs to be larger to reproduce the error, but for ImageNet,
        # a batch size as small as 4 reproduces the error
        self.dataset_train = datasets.MNIST(
            self.data_path, transform=transform, train=True, download=True
        )
        self.dataset_val = datasets.MNIST(
            self.data_path, transform=transform, train=False, download=True
        )

    def train_dataloader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,  # if either train or val num_workers=0, no error
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # if either train or val num_workers=0, no error
            pin_memory=True,
            drop_last=False,
        )


class PretrainModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(224 * 224, 384)
        self.head = nn.Linear(384, 128)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())

    def training_step(self, batch, _):
        images = batch[0]

        dummy = DictConfig({"test": 1})
        # this is problematic, but only if the attribute does not exist in dummy
        getattr(dummy, "anything", False)

        images = images.view(images.shape[0], -1)
        out1 = self.model(images)
        out2 = self.head(out1)  # not having self.head works fine
        loss = out2.sum()

        self.all_gather(out2)  # out1 works fine, otherwise problematic

        return loss

    def validation_step(self, batch, _):  # not having this works fine
        pass


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
