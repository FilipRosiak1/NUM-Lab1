import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

from data_module import EmojiDataModule
from model import EmojiClassifier


def train_model():
    # initialize wandb logger
    wandb_logger = WandbLogger(
        project="emoji-classifier",
        name="emoji-cnn-run",
        log_model=True
    )

    # initialize data module
    data_module = EmojiDataModule(
        data_dir="dataset",
        batch_size=32,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    )

    # initialize model
    model = EmojiClassifier(
        num_classes=18,
        lr=1e-3
    )

    # add checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='emoji-classifier-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3
    )

    # add early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10
    )

    # create trainer
    trainer = pl.Trainer(
        max_epochs=60,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        deterministic=True
    )

    # train model       
    trainer.fit(model, data_module)

    # test model
    trainer.test(model, data_module)

    # print best model path
    print("Best model saved at: ", checkpoint_callback.best_model_path)

    wandb.finish()

if __name__ == "__main__":
    train_model() 