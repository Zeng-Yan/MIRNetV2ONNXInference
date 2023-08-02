import cv2
import numpy as np
import onnxruntime
from pathlib import Path
from tqdm import tqdm
from loguru import logger

import torch


EXTENTIONS = ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG', '.bmp', '.BMP']


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)  # h, w, c
    return img


def save_img(filepath, img: np.ndarray):
    cv2.imwrite(filepath, cv2. cvtColor(img, cv2.COLOR_RGB2BGR))


class MIRNetV2Infer:
    def __init__(self) -> None:
        self.pad = [0, 0]

    def preprocessing(self, img: np.ndarray):
        img = img.transpose(2, 0, 1)  # c, h, w [0-255]
        img = np.expand_dims(img, 0)  # 1, c, h, w
        h, w = img.shape[2], img.shape[3]  # 1, c, h, w [0-255]
        pad_h = abs((h + 3) % 4 - 3)
        pad_w = abs((w + 3) % 4 - 3)
        self.pad = [pad_h, pad_w]

        img = img / 255.
        img = np.pad(img, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), 'reflect')
        return img.astype(np.float32)

    def postprocessing(self, output: np.ndarray):
        h, w = output.shape[2], output.shape[3]
        pad_h, pad_w = self.pad[0], self.pad[1]
        output = np.clip(output, 0, 1) * 255
        output = output[:, :, 0:h-pad_h, 0:w-pad_w]
        output = output[0].transpose(1, 2, 0)  # chw hwc
        return output

    def torch_infer(self, model, img, device='cuda'):
        model.eval()

        img = self.preprocessing(img)
        img = torch.from_numpy(img).float().to(device)
        with torch.no_grad():
            with torch.inference_mode():
                restored = model(img)
        restored = restored.cpu().detach().numpy()
        restored = self.postprocessing(restored)
        return restored

    def onnx_infer(self, img, onnx_path='mirnetv2.onnx'):
        sess = onnxruntime.InferenceSession(
            onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        img = self.preprocessing(img)
        restored = sess.run([output_name], {input_name: img})[0]
        restored = self.postprocessing(restored)
        return restored

    @staticmethod
    def export_onnx(model, name, verbose=False):
        input_names = ['input']
        output_names = ['output']
        dynamic_axes = {
            input_names[0]: {0: 'b', 1: 'c', 2: 'h', 3: 'w'},
            output_names[0]: {0: 'b', 1: 'c', 2: 'h', 3: 'w'}
        }
        inp = torch.ones((1, 3, 64, 64)).float()
        torch.onnx.export(
            model, inp, name, verbose=verbose,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        logger.info(f'Export ONNX to {name}')


def folder_processing(model, input_folder, output_folder):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    infer_helper = MIRNetV2Infer()
    files = [f for f in Path(input_folder).glob('*') if f.suffix in EXTENTIONS]
    output_folder = Path(output_folder)

    for img_file in tqdm(files):
        if torch.cuda.is_available():  # clear gpu cache
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        img = load_img(str(img_file))
        restored = infer_helper.torch_infer(model, img, device)
        # restored = infer_helper.onnx_infer(img)
        save_img(str(output_folder.joinpath(img_file.name)), restored)


if __name__ == "__main__":
    from mirnet_v2_arch import MIRNet_v2

    tasks = ['real_denoising', 'super_resolution',
             'contrast_enhancement', 'lowlight_enhancement']
    parameters = {
        'inp_channels': 3,
        'out_channels': 3,
        'n_feat': 80,
        'chan_factor': 1.5,
        'n_RRG': 4,
        'n_MRB': 2,
        'height': 3,
        'width': 2,
        'bias': False,
        'scale': 1,
        'task': tasks[2]
    }

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    mirnetv2 = MIRNet_v2(**parameters)
    checkpoint = torch.load('enhancement_fivek.pth')
    mirnetv2.load_state_dict(checkpoint['params'])

    # MIRNetV2Infer.export_onnx(mirnetv2, 'MIRNetV2.onnx', False)
    folder_processing(mirnetv2, 'images/ori_images', 'images/res_images')
