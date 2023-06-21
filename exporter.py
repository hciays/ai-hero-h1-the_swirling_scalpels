import os
import torch
from unet import UNet
import torch_tensorrt
from pathlib import Path
import time
import numpy as np


def benchmark(model,
              input_data,
              input_shape=(1, 1, 256, 256),
              dtype='fp32',
              nwarmup=50,
              nruns=1000):
    input_data = input_data.to("cuda")

    # if dtype=='fp16':
    #    input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            pred_loc = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print('Iteration %d/%d, avg batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))

    print("Input shape:", input_data.size())
    print('Average throughput: %.2f images/second' % (input_shape[0] / np.mean(timings)))
    return pred_loc


def export_model(version, imgsz, batch_size):
    path = "./lightning_logs/version_" + str(version) + "/checkpoints"
    shape = (batch_size, 1, imgsz, imgsz)
    if os.path.isdir(path):
        dir = list(Path(path).glob('*.ckpt'))
        if len(dir) > 1:
            raise Exception("Several checkpoints found, please specify one")
        elif len(dir) == 1:
            print(dir[0])
            model = UNet()
            checkpoint = torch.load(dir[0])
            model.load_state_dict(checkpoint['state_dict'])
            model.eval().to('cuda')
            tensor_rt_save_path = os.path.join(path, "trt_ts_module.ts")
            if not (os.path.exists(tensor_rt_save_path) and os.path.isfile(tensor_rt_save_path)):

                compiled_model = model.to_torchscript(file_path=os.path.join(path, "model.pt"), method="script",
                                                      example_inputs=shape)
                trt_model = torch_tensorrt.compile(compiled_model,
                                                   inputs=[torch_tensorrt.Input(
                                                    min_shape=(1, shape[1], shape[2], shape[3]),
                                                    opt_shape=shape,
                                                    max_shape=shape)],
                                                   enabled_precisions={torch.float, torch.half}
                                                   )
                torch.jit.save(trt_model, tensor_rt_save_path)
                print("TRT Model successfully saved under ", os.path.join(path, "trt_ts_module.ts"))
            else:
                trt_model = torch.jit.load(tensor_rt_save_path)
            benchmark(trt_model, input_data=torch.randn(shape), input_shape=shape, nruns=100)
        else:
            raise Exception("No checkpoint found")
    else:
        ValueError("Invalid version. Please check the version parameter")


if __name__ == '__main__':
    export_model(version=2, batch_size=64, imgsz=256)
