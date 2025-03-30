
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.optim import Adam, SGD
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger

import argparse
import csv
import random
import numpy as np

import dataset
from config import *
from model import XTiny_5branch as XTiny
from model import AudioExtractor

class LitModel(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

        self.dataset_name = args.dataset_name
        if self.dataset_name == 'esc50':
            self.num_classes=50        
        elif self.dataset_name == "urbansound8k":
            self.num_classes=10

        self.audio_extractor = AudioExtractor(n_time_masks=args.n_time_masks, 
                                              time_mask_param=args.time_mask_param, 
                                              n_freq_masks=args.n_freq_masks, 
                                              freq_mask_param=args.freq_mask_param)

        self.model = XTiny(num_classes=self.num_classes,
                           audio_extractor=self.audio_extractor,
                           use_segmentation=args.use_segmentation,
                           margin_ratio=args.margin_ratio)
        
        self.val_acc = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.num_classes,
        )
        self.train_acc=torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.num_classes,
        )
        self.k = 0
        self.label_smoothing = args.soft_epsilon
       
    def on_train_epoch_start(self):
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("learning_rate", current_lr, sync_dist=True)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        train_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(outputs, labels)
        self.log('train_loss', train_loss, sync_dist=True)
        self.log('train_acc', self.train_acc(outputs, labels), sync_dist=True)  
        return train_loss

    def on_train_epoch_end(self):
        pass


    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('val_loss', val_loss, sync_dist=True)  
        self.log('val_acc', self.val_acc(outputs, labels), sync_dist=True)
        return val_loss
    
    def on_validation_epoch_end(self):
        pass


    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = Adam(self.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-5)

        if self.args.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-30)
        elif self.args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    

    def forward(self, x):
        return self.model(x)


def train_kfold(model:LitModel, trainer:pl.Trainer, audio_ds:dataset.AudioDataset, k:int, args):
    '''
    Function to train the model for k-th fold
    args:
        trainer: pytorch_lightning.Trainer object
        audio_ds: Audio dataset object
        k: int, number of current fold
    '''
    # model re-initialization
    model.model.reset_parameters()
    model.k = k

    train_dl, val_dl = dataset.get_dataloader_kfold(dataset=audio_ds, k=k, args=args)
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    result = trainer.callback_metrics
    print(f'Fold {k} completed')
    return result


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument('--root', type=str, default='/workspace/CNNAudioClassification/data')
    parser.add_argument('--dataset_name', type=str, default='esc50')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--soft_epsilon', type=float, default=0)

    # Data Augmentation Arguments
    parser.add_argument('--n_time_masks', type=int, default=None)
    parser.add_argument('--n_freq_masks', type=int, default=None)
    parser.add_argument('--time_mask_param', type=int, default=None)
    parser.add_argument('--freq_mask_param', type=int, default=None)

    # Model Arguments
    parser.add_argument('--use_segmentation', action='store_true')
    parser.add_argument('--model_name', type=str, default='1branch')
    parser.add_argument('--seed', type=int, default=777777)
    parser.add_argument('--out_dir', type=str, default='checkpoints')
    parser.add_argument('--margin_ratio', type=int, default=0)

    # Training Arguments
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=300)

    parser.add_argument('--gpu_id', type=int, default=0)

    return parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)

def main(args):
    seed_everything(args.seed)

    model = LitModel(args)
    dirpath = args.out_dir

    if args.dataset_name == 'esc50':
        audio_ds = dataset.ESC50(
            root = args.root,
            args = args,
            download = True,
            dataset_name = args.dataset_name
        )
    elif args.dataset_name == 'urbansound8k':
        audio_ds = dataset.UrbanSound8k(
            root = args.root,
            args = args,
            download = True,
            dataset_name = 'urbansound8k'
        )
    
    k = audio_ds.k

    fold_results = []
    for k_i in range(1, k+1):
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            dirpath=dirpath,
            filename=args.dataset_name + f"-fold{k_i}-" + "{epoch}-{val_acc:.04f}",
        )
        swa_callback = StochasticWeightAveraging(swa_lrs=args.lr/10)
        call_backs = [checkpoint_callback, swa_callback]
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            callbacks=call_backs,
            accelerator='gpu', devices=[args.gpu_id],
        )
        result = train_kfold(model=model, trainer=trainer, audio_ds=audio_ds, k=k_i, args=args)
        result['fold'] = k_i
        fold_results.append(result)

    with open(f'fold_results_{args.dataset_name}_margin{args.margin_ratio}.csv', mode='w', newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fold_results[0].keys())
        writer.writeheader()
        for fold_result in fold_results:
            writer.writerow(fold_result)

    print('Training Completed')


if __name__ == "__main__":
    args = parse_args()
    main(args)