import cv2
import numpy as np
import onnxruntime
from pathlib import Path
from tqdm import tqdm

EXTENTIONS = ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG', '.bmp', '.BMP']


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)  # h, w, c
    return img


def save_img(filepath, img: np.ndarray):
    cv2.imwrite(filepath, cv2. cvtColor(img, cv2.COLOR_RGB2BGR))


class MIRNetV2ONNXInfer:
    def __init__(self, onnx_path='MIRNetV2.onnx') -> None:
        self.pad = [0, 0]

        self.sess = onnxruntime.InferenceSession(
            onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

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

    def onnx_infer(self, img):
        img = self.preprocessing(img)
        restored = self.sess.run([self.output_name], {self.input_name: img})[0]
        restored = self.postprocessing(restored)
        return restored


def folder_processing(onnx_path, input_folder, output_folder):
    infer_helper = MIRNetV2ONNXInfer(onnx_path)
    files = [f for f in Path(input_folder).glob('*') if f.suffix in EXTENTIONS]
    output_folder = Path(output_folder)

    for img_file in tqdm(files):
        img = load_img(str(img_file))
        restored = infer_helper.onnx_infer(img)
        save_img(str(output_folder.joinpath(img_file.name)), restored)


if __name__ == "__main__":
    folder_processing('MIRNetV2.onnx', 'images/ori_images',
                      'images/res_images')
