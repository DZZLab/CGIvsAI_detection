import os
import shutil
import torch
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as nnFunc
from torchmetrics import Accuracy, Precision
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Needed to implement memory mapping, as it was trying to load entire .npy to RAM. 
class MemoryMappedDataset(Dataset):
    def __init__(self, video_path, labels_path):
        self.video_path = video_path
        self.labels_path = labels_path
        self.labels = np.load(labels_path, mmap_mode='r+')
        self.data = np.load(video_path, mmap_mode='r+')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video = torch.tensor(self.data[idx], dtype=torch.float32).permute(3, 0, 1, 2)  # C, T, H, W - tensor dimensions
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"video": video, "label": label}

#Data module for loading my Preprocessed data.
class VideoDataModule(pl.LightningDataModule):
    def __init__(self, train_video_path, train_labels_path, test_video_path, test_labels_path, batch_size=1, num_workers=2):
        super().__init__()
        self.train_video_path = train_video_path
        self.train_labels_path = train_labels_path
        self.test_video_path = test_video_path
        self.test_labels_path = test_labels_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    #Unused as I am manually preprocessing my data as of now. 
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'train' or stage is None:
            dir_tr, nm_tr = os.path.split(self.train_labels_path)
            print('Training Dataset located in the \'{}\' folder.'.format(dir_tr))
            self.train_dataset = MemoryMappedDataset(self.train_video_path, self.train_labels_path)
            self.val_dataset = MemoryMappedDataset(self.train_video_path, self.train_labels_path)
            self.test_dataset = MemoryMappedDataset(self.train_video_path, self.train_labels_path)
        elif stage == 'test':
            dir_te, nm_te = os.path.split(self.test_labels_path)
            dir_tr, nm_tr = os.path.split(self.train_labels_path)
            print('Training Dataset located in the \'{}\' folder.'.format(dir_tr))
            self.train_dataset = MemoryMappedDataset(self.train_video_path, self.train_labels_path)
            self.val_dataset = MemoryMappedDataset(self.train_video_path, self.train_labels_path)
            print('Testing Dataset located in the \'{}\' folder.'.format(dir_te))
            self.test_dataset = MemoryMappedDataset(self.test_video_path, self.test_labels_path)

        # self.train_dataset = MemoryMappedDataset(self.video_path, self.labels_path)
        # self.val_dataset = MemoryMappedDataset(self.video_path, self.labels_path)
        # self.test_dataset = MemoryMappedDataset(self.video_path, self.labels_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

#3D convolution neural network
class CNN3D(pl.LightningModule):
    def __init__(self, input_shape):
        super(CNN3D, self).__init__()
        self.model = torch.nn.Sequential(
            #The input to output ratio may be bad, but this is what my GPU is just barely capable of running.
            torch.nn.Conv3d(3, 12, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            torch.nn.Conv3d(12, 20, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            #into Rectified Linear Unit activation function - rectifies values
            torch.nn.ReLU(),
            torch.nn.MaxPool3d((2, 2, 2)),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d((2, 2, 2)),
            torch.nn.Flatten()
        )
        self.accuracy = Accuracy()
        self.precision = Precision(num_classes=2, average='macro')
        self.output_size = self.model(torch.zeros(1, *input_shape)).numel()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.output_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch['video'], batch['label']
        outputs = self(inputs)
        loss = nnFunc.cross_entropy(outputs, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_learning_rate()
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['video'], batch['label']
        outputs = self(inputs)
        val_loss = nnFunc.cross_entropy(outputs, labels)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch['video'], batch['label']
        outputs = self(inputs)
        loss = nnFunc.cross_entropy(outputs, labels)

        #Calculations for accuracy and precision
        preds = torch.argmax(outputs, dim=1)
        acc = self.accuracy(preds, labels)
        prec = self.precision(preds, labels)

        #Logging in terminal
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_precision', prec, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {"test_loss": loss, "test_accuracy": acc, "test_precision": prec}

    def log_learning_rate(self):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        #MARK: Learning Rate
        '''
        Added in a learning rate scheduler 'ReduceLROnPlateau' b/c I was noticing the val_loss and train_loss per epoch 
        would overshoot. I.e, it would gradually decrease to what seems like almost a minimum then spikes back up.
        This ReduceLROnPlateu should reduces the learning rate by a factor when the loss values are slowing down, 
        to prevent overshooting when a minimum in the gradient is reached.
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=4)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss"}

    def on_train_epoch_end(self, unused=None):
        super().on_train_epoch_end()
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        if current_lr < 1e-8:
            self.reset_learning_rate()

    def reset_learning_rate(self):
        new_lr = 0.00001
        #Before Resetting, make a copy of the last checkpoint.
        filep = 'Checkpoints/last.ckpt'
        destination = 'Checkpoints/Backup/last.ckpt'
        shutil.copy(filep, destination)
        for param_group in self.trainer.optimizers[0].param_groups:
            param_group['lr'] = new_lr
        print(f"Reset learning rate to {new_lr}")


def main():
    train_video_path = 'data/VideoData.npy'
    train_labels_path = 'data/VideoLabels.npy'
    test_video_path = 'TestingData/VideoData.npy'
    test_labels_path = 'TestingData/VideoLabels.npy'

    video_data_module = VideoDataModule(
        train_video_path=train_video_path,
        train_labels_path=train_labels_path,
        test_video_path=test_video_path,
        test_labels_path=test_labels_path
    )

    video_data_module.setup(stage='test')

    checkpoint_dir = './Checkpoints/'
    #Checkpoint name to resume from. [Must include '.ckpt' at end]training 
    checkpoint_filename = 'last.ckpt'#'last.ckpt'
    #Checkpoint name to save. [Must leave out '.ckpt' at the end]
    checkpoint_name = 'BestCkpt'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', dirpath=checkpoint_dir, filename=checkpoint_name,
        save_top_k=1, mode='min', save_last=True, auto_insert_metric_name=False,
        enable_version_counter=False, verbose=True
    )

    ckpt_path = os.path.join(checkpoint_dir, checkpoint_filename) if os.path.exists(checkpoint_path) else None

    sample_batch = next(iter(video_data_module.train_dataloader()))
    input_shape = sample_batch['video'].shape[1:]
    model = CNN3D(input_shape=input_shape)

    trainer = Trainer(
        max_epochs=1033, accelerator='auto', devices=1, precision='16-mixed',
        callbacks=[checkpoint_callback], logger=True
    )

    trainer.fit(model, video_data_module, ckpt_path=ckpt_path)
    trainer.test(model, video_data_module.test_dataloader())  # Running the test after training

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Model trained for {:.2f} seconds".format(time.time() - start_time))
