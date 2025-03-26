from torch.utils.data import DataLoader, ConcatDataset

def get_dataloader_kfold(dataset, k, args):
    folds = dataset.folds
    K = len(folds)
    train_set = ConcatDataset([folds[i] for i in range(1, K) if i != k])
    val_set = folds[k-1]

    train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_dl = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    return train_dl, val_dl