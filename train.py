import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import vision_transformer as vits
from torchvision import datasets, transforms


class DataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

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
            self.hparams.data_path, transform=transform, train=True, download=True
        )
        self.dataset_val = datasets.MNIST(
            self.hparams.data_path, transform=transform, train=False, download=True
        )

    def train_dataloader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,  # if either train or val num_workers=0, no error
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,  # if either train or val num_workers=0, no error
            pin_memory=True,
            drop_last=False,
        )


class PretrainModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = vits.vit_small(**self.hparams.model_args)
        self.head = nn.Linear(self.model.num_features, self.hparams.out_dim)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())

    def training_step(self, batch, _):
        images = batch[0]

        # this is problematic, but only if the attribute does not exist in model_args
        getattr(self.hparams.model_args, "anything", False)

        out1 = self.model(images)[0]
        out2 = self.head(out1)  # not having self.head works fine
        loss = out2.sum()

        self.all_gather(out2)  # out1 works fine, otherwise problematic

        return loss

    def validation_step(self, batch, _):  # not having this works fine
        pass


@hydra.main(config_path="config", config_name="simple_test")
def main(config):
    datamodule = DataModule(**config.datamodule)
    module = PretrainModule(**config.taskmodule, epochs=config.trainer.max_epochs)
    trainer = pl.Trainer(**config.trainer)
    trainer.fit(
        module,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()
