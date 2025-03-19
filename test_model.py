from model import EmojiClassifier
from data_module import EmojiDataModule
import pytorch_lightning as pl

if __name__ == "__main__":
    model = EmojiClassifier.load_from_checkpoint("checkpoints/best_model.ckpt")
    data_module = EmojiDataModule("dataset")
    data_module.setup()

    trainer = pl.Trainer(accelerator='auto', devices=1)
    trainer.test(model, data_module)
