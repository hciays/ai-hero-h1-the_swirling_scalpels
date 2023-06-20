import torch
from dataset import CellDataset, val_transform
from unet import UNet
from argparse import ArgumentParser
from utils_img import post
from multiprocessing import Pool
from unet import benchmark, predict_instance_n


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
    parser.add_argument("--num_cpus", type=int, default=12)
    parser.add_argument("--trt_path", type=str, default="")

    args = parser.parse_args()

    device = torch.device("cuda")
    root_dir = args.root_dir
    pred_dir = args.pred_dir
    split = args.split
    model = None

    instance_seg_val_data = CellDataset(root_dir, split=split, transform=val_transform(), border_core=False)
    instance_seg_valloader = torch.utils.data.DataLoader(
        instance_seg_val_data, batch_size=16, shuffle=False, num_workers=12
    )
    if args.trt_path != "":
        model = torch.jit.load(args.trt_path)
        files_names, pre_pred = [], []
        for batch, _, _, file_name in instance_seg_valloader:
            pred = torch.argmax(benchmark(model=model, input_data=batch), 1)
            files_names.append(file_name)
            pre_pred.append(pred)
        pre = None
        with Pool(args.num_cpus) as p:
            pre = p.map(post, pred)
        # predict instances and save them in the pred_dir
        predict_instance_n(batch_n=files_names, preds=pre, pred_dir=pred_dir)

    else:
        model = UNet()
        checkpoint = torch.load(args.from_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        # predict instances and save them in the pred_dir
        model.predict_instance_segmentation_from_border_core(instance_seg_valloader, pred_dir=pred_dir)


