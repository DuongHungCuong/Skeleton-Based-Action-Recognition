import cv2
import os
import time
import numpy as np
from camerea_loader.cameraloader import CamLoader, CamLoader_Q
from detection_keypoint.kapao import KapaoTRT
from tracking.sort_w_keypoints import Sort
from action_recognition.stgcn import TSSTG
from miscellaneous_utils.mis_utils import load_config

config_path = '../configs/configs.yml'
configs = load_config(config_path)

kp_model_path = configs['detection']['model_path']
cfg_path = configs['detection']['cfg_path']
device = configs['device']
kapaoTrt = KapaoTRT(kp_model_path, cfg_path, device)
person_tracker = Sort()
stgcn_model_path = configs['action_recognition']['model_path']
action_model = TSSTG(stgcn_model_path, device)

cam_source = configs['video_path']
output_video_path = configs['output_video_path']

if type(cam_source) is str and os.path.isfile(cam_source):
	cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=None).start()
else:
	cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source, preprocess=None).start()

fps = cam.fps
size = cam.frame_size
writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size, isColor=True,)

cnt = 0
n1 = 0 
fps1 = 0
n2 = 0 
fps2 = 0

while cam.grabbed():
	cnt += 1
	img = cam.getitem()
	print(f'shape img: {img.shape}')
	st = time.time()
	bboxes, poses = kapaoTrt.detect(img)
	fps1 += 1/(time.time()-st)
	n1 += 1
	print(f'FPS kapao: {1/(time.time()-st)}')
	print(f'len: {len(poses)}')
	tracks, keypoints_list = person_tracker.update(np.asarray(bboxes), np.asarray(poses))
	for track, keypoints in zip(tracks, keypoints_list):
		print(f'For id: {int(track[9])}, {len(keypoints)}')
		cv2.rectangle(img, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), (0, 255, 0), 2)
		cv2.putText(img, str(int(track[9])), (int(track[0]), int(track[1]-40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 69, 255), 3)
		if len(keypoints) == 30:
			# pts = np.stack(keypoints, axis=0)
			pts = np.array(keypoints, dtype=np.float32)
			print(f'pts: {pts.shape}')
			st = time.time()
			out = action_model.predict(pts, img.shape[:2])
			fps2 += 1/(time.time()-st)
			n2 += 1
			print(f'FPS STGCN: {1/(time.time()-st)}')
			action_name = action_model.class_names[out[0].argmax()]
			print(f'action_name: {action_name}')
			cv2.putText(img, action_name, (int(track[0]+25), int(track[1]-40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 69, 255), 3)
	if len(poses) > 0:
		img = kapaoTrt.visualize(img, bboxes, poses)
	else:
		print('EMPTY DETECTION')
	writer.write(img)

cam.stop()
writer.release()

print(f'MEAN FPS KAPAO: {fps1/n1}')
print(f'MEAN FPS STGCN: {fps2/n2}')