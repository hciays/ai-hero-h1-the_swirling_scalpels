import os
import torch
from unet import UNet
import torch_tensorrt
from pathlib import Path


def export_batch_size():
    return 128


def export_number_channels():
    return 1


def export_model(version):
    path = "./lightning_logs/version_" + str(version) + "/checkpoints"
    if os.path.isdir(path):
        dir = os.listdir(path)
        if len(dir) > 1:
            raise Exception("Several checkpoints found, please specify one")
        elif len(dir) == 1:
            print(os.path.join(path, dir[0]))
            model = UNet().eval().to('cuda')
            checkpoint = torch.load(os.path.join(path,dir[0]))
            model.load_state_dict(checkpoint['state_dict'])
            if not Path("trt_ts_module.ts").exists():
                compiled_model = model.to_torchscript(file_path=os.path.join(path,"model.pt"), method="script",
                                                      example_inputs=(export_batch_size(),
                                                                      export_number_channels(),
                                                                      256, 256))
                trt_model = torch_tensorrt.compile(compiled_model,
                                                   inputs=[torch_tensorrt.Input((export_batch_size(),
                                                                                 export_number_channels(),
                                                                                 256,
                                                                                 256))],
                                                   enabled_precisions={torch.float, torch.half}
                                                   )
                torch.jit.save(trt_model, os.path.join(path,"trt_ts_module.ts"))
                print("TRT Model successfully saved under" + os.path.join(path,"trt_ts_module.ts"))

            return
        else:
            raise Exception("No checkpoint found")
    else:
        ValueError("Invalid version. Please check the version parameter")
