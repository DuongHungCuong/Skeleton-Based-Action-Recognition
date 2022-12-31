import cv2
import torch
import random
import time
import yaml
import numpy as np
import tensorrt as trt
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
from .utils.augmentations import letterbox
from .val import run_nms, post_process_batch


class KapaoTRT():
    def __init__(self, model_path, cfg_path, device='cuda:0'):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        self.bindings = OrderedDict()
        for index in range(self.model.num_bindings):
            name = self.model.get_binding_name(index)
            dtype = trt.nptype(self.model.get_binding_dtype(index))
            shape = tuple(self.model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.model.create_execution_context()

        self.device = torch.device(device)

        with open(cfg_path) as f:
            self.data = yaml.safe_load(f)  # load data dict
        self.data['imgsz'] = 1280
        self.data['conf_thres'] = 0.2
        self.data['iou_thres'] = 0.45
        self.data['use_kp_dets'] = True
        self.data['conf_thres_kp'] = 0.05
        self.data['iou_thres_kp'] = 0.4
        self.data['conf_thres_kp_person'] = 0.1
        self.data['overwrite_tol'] = 50
        self.data['scales'] = 1
        self.data['flips'] = None
        self.data['count_fused'] = False

        # warmup for 10 times
        for _ in range(10):
            tmp = torch.randn(1,3,self.data['imgsz'],self.data['imgsz']).to(self.device)
            self.binding_addrs['images'] = int(tmp.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))

    def _preprocess(self, image):
        image, ratio, dwdh = letterbox(image, new_shape=(self.data['imgsz'], self.data['imgsz']), auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im = torch.from_numpy(im).to(self.device)
        im /= 255

        return im

    def _inference(self, im):
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        out = self.bindings['output'].data

        return out

    def _postprocess(self, out, im, org_shape):
        st = time.time()
        person_dets, kp_dets = run_nms(self.data, out)
        print(f'FPS post1 kapao: {1/(time.time()-st)}')
        st = time.time()
        bboxes, poses, _, _, _ = post_process_batch(self.data, im, [], [[org_shape]], person_dets, kp_dets)
        print(f'FPS post2 kapao: {1/(time.time()-st)}')

        return bboxes, poses

    def detect(self, img):
        import time
        st = time.time()
        im = self._preprocess(img)
        print(f'FPS pre kapao: {1/(time.time()-st)}')
        org_shape = img.shape[:2]
        st = time.time()
        out = self._inference(im)
        print(f'FPS infer kapao: {1/(time.time()-st)}')
        st = time.time()
        bboxes, poses = self._postprocess(out, im, org_shape)
        print(f'FPS post kapao: {1/(time.time()-st)}')

        return bboxes, poses

    def visualize(self, img, bboxes, poses):
        for j, (bbox, pose) in enumerate(zip(bboxes, poses)):
            x1, y1, x2, y2 = bbox
            # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=2)
            for x, y, c in pose[self.data['kp_face']]:
                cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 3)
            for seg in self.data['segments'].values():
                if (pose[seg[0], -1] and pose[seg[1], -1]):
                    pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
                    pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
                    cv2.line(img, pt1, pt2, (0, 69, 255), 2)

        return img




            