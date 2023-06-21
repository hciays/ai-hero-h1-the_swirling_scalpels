from argparse import ArgumentParser
import torch
import torch._dynamo
import pytorch_lightning as pl

from dataset import CellDataset, train_transform, val_transform
from unet import UNet

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/hkfs/work/workspace/scratch/hgf_pdv3669-health_train_data/train",
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--local_test', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--reproducibility', type=bool, default=True)
    args = parser.parse_args()

    root_dir = args.root_dir
    print(root_dir)

    # Data
    train_data = CellDataset(root_dir, split="train", transform=train_transform(), local_test=args.local_test)
    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_data = CellDataset(root_dir, split="val", transform=val_transform(), local_test=args.local_test)
    valloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    # Initialize the model and trainer
    model = UNet()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_iou',
        mode='max',
        filename='best_model',
        save_top_k=1,
    )
    if args.gpus == 1:
        trainer = pl.Trainer(accelerator=args.device,
                             devices=args.gpus,
                             max_epochs=args.num_epochs,
                             precision="16-mixed",
                             benchmark=True,
                             callbacks=checkpoint_callback)
        #opt_model = torch.compile(model)
        # Train the model
        trainer.fit(model, trainloader, valloader)
    else:
        trainer = pl.Trainer(accelerator=args.device,
                             max_epochs=args.num_epochs,
                             precision="16-mixed",
                             benchmark=True,
                             devices=args.gpus,
                             strategy="ddp",
                             num_nodes=1)
        # Train the model
        trainer.fit(model, trainloader, valloader)
