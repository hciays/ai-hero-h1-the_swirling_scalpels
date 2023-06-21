import torch
from dataset import CellDataset, val_transform
from unet import UNet
from argparse import ArgumentParser
from utils_img import post
from multiprocessing import Pool
import tifffile
import os
import cv2
from acvl_utils.instance_segmentation.instance_as_semantic_seg import convert_semantic_to_instanceseg
import numpy as np
import torch_tensorrt


def predict_instance_n(predictions):
    file_name, preds, pred_dir = predictions

    preds = torch.argmax(preds, 1)
    for i in range(preds.shape[0]):
        # convert to instance segmentation
        print(np.max(preds[i].numpy()).astype(np.uint8))
        instance_segmentation = convert_semantic_to_instanceseg(
            np.array(preds[i].numpy()).astype(np.uint8),
            spacing=(1, 1, 1),
            isolated_border_as_separate_instance_threshold=15,
            small_center_threshold=30).squeeze()
        # instance_segmentation = preds[i].astype(np.uint8)
        # resize to size 256x256
        resized_instance_segmentation = cv2.resize(instance_segmentation.astype(np.float32), (256, 256),
                                                   interpolation=cv2.INTER_NEAREST)
        # save file
        save_dir, save_name = os.path.join(pred_dir, file_name[i].split('/')[0]), file_name[i].split('/')[1]
        os.makedirs(save_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(save_dir, save_name.replace('.tif', '_256.tif')),
                         resized_instance_segmentation.astype(np.uint8))


def test(predictions):
    file_name, preds = predictions
    print(len(file_name), preds.shape)
    return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/hkfs/work/workspace/scratch/hgf_pdv3669-health_train_data/train",
    )
    parser.add_argument("--from_checkpoint", type=str,
                        default='./lightning_logs/version_0/checkpoints/epoch=99-step=10000.ckpt')
    parser.add_argument("--pred_dir", default='./pred')
    parser.add_argument("--split", default="val", help="val=sequence c")
    parser.add_argument("--num_cpus", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    device = torch.device("cuda")
    root_dir = args.root_dir
    pred_dir = args.pred_dir
    split = args.split
    model = None

    instance_seg_val_data = CellDataset(root_dir, split=split, transform=val_transform(imgsz=args.imgsz),
                                        border_core=False, local_test=True)
    instance_seg_valloader = torch.utils.data.DataLoader(
        instance_seg_val_data, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=True
    )
    if args.from_checkpoint.endswith('.ts'):
        model = torch.jit.load(args.from_checkpoint)
        model.eval().to('cuda')
        preds = []
        with torch.no_grad():
            for batch, _, _, file_name in instance_seg_valloader:
                print(batch.shape)
                batch = batch.to('cuda')
                preds.append((file_name, model(batch).detach().cpu(), pred_dir))
        del model
        pre = None
        with Pool(len(preds)) as p:
            pre = p.map(predict_instance_n, preds)
        # predict instances and save them in the pred_dir

    elif args.from_checkpoint.endswith('.ckpt'):
        model = UNet()
        checkpoint = torch.load(args.from_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        # predict instances and save them in the pred_dir
        model.predict_instance_segmentation_from_border_core(instance_seg_valloader, pred_dir=pred_dir)
    else:
        raise Exception("No checkpoint nor Tensorrt found")
