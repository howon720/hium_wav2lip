# from os import listdir, path
# import numpy as np
# import scipy, cv2, os, sys, argparse, audio
# import json, subprocess, random, string
# from tqdm import tqdm
# from glob import glob
# import torch, face_detection
# from models import Wav2Lip
# import platform

# parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

# parser.add_argument('--checkpoint_path', type=str, 
# 					help='Name of saved checkpoint to load weights from', required=True)

# parser.add_argument('--face', type=str, 
# 					help='Filepath of video/image that contains faces to use', required=True)
# parser.add_argument('--audio', type=str, 
# 					help='Filepath of video/audio file to use as raw audio source', required=True)
# parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
# 								default='results/result_voice.mp4')

# parser.add_argument('--static', type=bool, 
# 					help='If True, then use only first video frame for inference', default=False)
# parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
# 					default=25., required=False)

# parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
# 					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

# parser.add_argument('--face_det_batch_size', type=int, 
# 					help='Batch size for face detection', default=16)
# parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

# parser.add_argument('--resize_factor', default=1, type=int, 
# 			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

# parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
# 					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
# 					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

# parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
# 					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
# 					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

# parser.add_argument('--rotate', default=False, action='store_true',
# 					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
# 					'Use if you get a flipped result, despite feeding a normal looking video')

# parser.add_argument('--nosmooth', default=False, action='store_true',
# 					help='Prevent smoothing face detections over a short temporal window')

# args = parser.parse_args()
# args.img_size = 96

# if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
# 	args.static = True

# def get_smoothened_boxes(boxes, T):
# 	for i in range(len(boxes)):
# 		if i + T > len(boxes):
# 			window = boxes[len(boxes) - T:]
# 		else:
# 			window = boxes[i : i + T]
# 		boxes[i] = np.mean(window, axis=0)
# 	return boxes

# def face_detect(images):
# 	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
# 											flip_input=False, device=device)

# 	batch_size = args.face_det_batch_size
	
# 	while 1:
# 		predictions = []
# 		try:
# 			for i in tqdm(range(0, len(images), batch_size)):
# 				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
# 		except RuntimeError:
# 			if batch_size == 1: 
# 				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
# 			batch_size //= 2
# 			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
# 			continue
# 		break

# 	results = []
# 	pady1, pady2, padx1, padx2 = args.pads
# 	for rect, image in zip(predictions, images):
# 		if rect is None:
# 			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
# 			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

# 		y1 = max(0, rect[1] - pady1)
# 		y2 = min(image.shape[0], rect[3] + pady2)
# 		x1 = max(0, rect[0] - padx1)
# 		x2 = min(image.shape[1], rect[2] + padx2)
		
# 		results.append([x1, y1, x2, y2])

# 	boxes = np.array(results)
# 	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
# 	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

# 	del detector
# 	return results 

# def datagen(frames, mels):
# 	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

# 	if args.box[0] == -1:
# 		if not args.static:
# 			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
# 		else:
# 			face_det_results = face_detect([frames[0]])
# 	else:
# 		print('Using the specified bounding box instead of face detection...')
# 		y1, y2, x1, x2 = args.box
# 		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

# 	for i, m in enumerate(mels):
# 		idx = 0 if args.static else i%len(frames)
# 		frame_to_save = frames[idx].copy()
# 		face, coords = face_det_results[idx].copy()

# 		face = cv2.resize(face, (args.img_size, args.img_size))
			
# 		img_batch.append(face)
# 		mel_batch.append(m)
# 		frame_batch.append(frame_to_save)
# 		coords_batch.append(coords)

# 		if len(img_batch) >= args.wav2lip_batch_size:
# 			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

# 			img_masked = img_batch.copy()
# 			img_masked[:, args.img_size//2:] = 0

# 			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
# 			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

# 			yield img_batch, mel_batch, frame_batch, coords_batch
# 			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

# 	if len(img_batch) > 0:
# 		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

# 		img_masked = img_batch.copy()
# 		img_masked[:, args.img_size//2:] = 0

# 		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
# 		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

# 		yield img_batch, mel_batch, frame_batch, coords_batch

# mel_step_size = 16
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using {} for inference.'.format(device))

# def _load(checkpoint_path):
# 	if device == 'cuda':
# 		checkpoint = torch.load(checkpoint_path)
# 	else:
# 		checkpoint = torch.load(checkpoint_path,
# 								map_location=lambda storage, loc: storage)
# 	return checkpoint

# def load_model(path):
# 	model = Wav2Lip()
# 	print("Load checkpoint from: {}".format(path))
# 	checkpoint = _load(path)
# 	s = checkpoint["state_dict"]
# 	new_s = {}
# 	for k, v in s.items():
# 		new_s[k.replace('module.', '')] = v
# 	model.load_state_dict(new_s)

# 	model = model.to(device)
# 	return model.eval()

# def main():
# 	if not os.path.isfile(args.face):
# 		raise ValueError('--face argument must be a valid path to video/image file')

# 	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
# 		full_frames = [cv2.imread(args.face)]
# 		fps = args.fps

# 	else:
# 		video_stream = cv2.VideoCapture(args.face)
# 		fps = video_stream.get(cv2.CAP_PROP_FPS)

# 		print('Reading video frames...')

# 		full_frames = []
# 		while 1:
# 			still_reading, frame = video_stream.read()
# 			if not still_reading:
# 				video_stream.release()
# 				break
# 			if args.resize_factor > 1:
# 				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

# 			if args.rotate:
# 				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

# 			y1, y2, x1, x2 = args.crop
# 			if x2 == -1: x2 = frame.shape[1]
# 			if y2 == -1: y2 = frame.shape[0]

# 			frame = frame[y1:y2, x1:x2]

# 			full_frames.append(frame)

# 	print ("Number of frames available for inference: "+str(len(full_frames)))

# 	if not args.audio.endswith('.wav'):
# 		print('Extracting raw audio...')
# 		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

# 		subprocess.call(command, shell=True)
# 		args.audio = 'temp/temp.wav'

# 	wav = audio.load_wav(args.audio, 16000)
# 	mel = audio.melspectrogram(wav)
# 	print(mel.shape)

# 	if np.isnan(mel.reshape(-1)).sum() > 0:
# 		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

# 	mel_chunks = []
# 	mel_idx_multiplier = 80./fps 
# 	i = 0
# 	while 1:
# 		start_idx = int(i * mel_idx_multiplier)
# 		if start_idx + mel_step_size > len(mel[0]):
# 			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
# 			break
# 		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
# 		i += 1

# 	print("Length of mel chunks: {}".format(len(mel_chunks)))

# 	full_frames = full_frames[:len(mel_chunks)]

# 	batch_size = args.wav2lip_batch_size
# 	gen = datagen(full_frames.copy(), mel_chunks)

# 	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
# 											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
# 		if i == 0:
# 			model = load_model(args.checkpoint_path)
# 			print ("Model loaded")

# 			frame_h, frame_w = full_frames[0].shape[:-1]
# 			out = cv2.VideoWriter('temp/result.avi', 
# 									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

# 		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
# 		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

# 		with torch.no_grad():
# 			pred = model(mel_batch, img_batch)

# 		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
# 		for p, f, c in zip(pred, frames, coords):
# 			y1, y2, x1, x2 = c
# 			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

# 			f[y1:y2, x1:x2] = p
# 			out.write(f)

# 	out.release()

# 	# command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)

# 	command = (
# 	    f'ffmpeg -y '
# 	    '-i temp/result.avi '
# 		f'-i {args.audio} '
#    		'-c:v libx264 -crf 16 -preset slow '   # crf 값 내리면 화질 올라가는 듯 
#     	'-c:a aac -b:a 192k '
#     	'-vf "scale=iw:ih" '
#     	f'{args.outfile}'
#  	)
# 	subprocess.call(command, shell=platform.system() != 'Windows')

# if __name__ == '__main__':
# 	main()


import os
import sys
import json
import random
import string
import subprocess
import platform
from glob import glob
from os import listdir, path

import numpy as np
import scipy
import cv2
import torch
from tqdm import tqdm

import audio  # Wav2Lip 오디오 유틸
import face_detection  # 얼굴 검출 라이브러리
from models import Wav2Lip  # Wav2Lip 모델 클래스

import argparse

# ----------------------------------------
# 1. 커맨드라인 인자 정의
# ----------------------------------------
parser = argparse.ArgumentParser(
    description='Wav2Lip 모델을 사용한 영상 립싱크 추론 스크립트')
parser.add_argument('--checkpoint_path', type=str, required=True,
                    help='모델 체크포인트 경로')
parser.add_argument('--face', type=str, required=True,
                    help='입력 얼굴 이미지/영상 파일 경로')
parser.add_argument('--audio', type=str, required=True,
                    help='입력 오디오/영상 파일 경로')
parser.add_argument('--outfile', type=str, default='results/result_voice.mp4',
                    help='출력 영상 파일 경로')
# 이미지 고정 모드, fps 지정, 얼굴 패딩 설정 등 추가 옵션
parser.add_argument('--static', type=bool, default=False,
                    help='True면 첫 프레임만 사용')
parser.add_argument('--fps', type=float, default=25.,
                    help='정적 이미지 입력 시 FPS')
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='face 패딩: [top, bottom, left, right]')
parser.add_argument('--face_det_batch_size', type=int, default=16,
                    help='얼굴 검출 배치 크기')
parser.add_argument('--wav2lip_batch_size', type=int, default=128,
                    help='Wav2Lip 모델 배치 크기')
parser.add_argument('--resize_factor', type=int, default=1,
                    help='프레임 리사이즈 비율')
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                    help='영상 자르기: [top, bottom, left, right]')
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                    help='고정 바운딩 박스 사용 시 지정 [top, bottom, left, right]')
parser.add_argument('--rotate', action='store_true', default=False,
                    help='True면 영상 90도 회전')
parser.add_argument('--nosmooth', action='store_true', default=False,
                    help='True면 얼굴 박스 스무딩 비활성화')
args = parser.parse_args()
args.img_size = 96  # 모델 입력 이미지 크기 (96x96)

# 입력이 이미지 파일이면 static 모드 자동 설정
if os.path.isfile(args.face) and args.face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
    args.static = True


#  얼굴 박스 스무딩 함수
def get_smoothened_boxes(boxes, T=5):
    """
    시간축으로 박스 위치를 평균 내어 흔들림 완화
    """
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

#  얼굴 검출 함수
def face_detect(images):
    """
    이미지 리스트에서 얼굴 영역 검출 후 반환
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D,
        flip_input=False,
        device=device)
    batch_size = args.face_det_batch_size

    # 배치 단위로 얼굴 검출
    while True:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                batch = np.array(images[i: i + batch_size])
                predictions.extend(
                    detector.get_detections_for_batch(batch)
                )
        except RuntimeError:
            # OOM 발생 시 배치 크기 절반으로 재시도
            if batch_size == 1:
                raise RuntimeError(
                    '이미지가 너무 커서 GPU에서 얼굴 검출 불가.')
            batch_size //= 2
            print(f'GPU OOM, 배치 크기 감소: {batch_size}')
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError('얼굴을 찾을 수 없습니다!')
        # 패딩 적용하여 좌표 계산
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        results.append([x1, y1, x2, y2])
    boxes = np.array(results)
    if not args.nosmooth:
        boxes = get_smoothened_boxes(boxes)

    # (face_img, (y1,y2,x1,x2)) 리스트 반환
    return [[image[y1:y2, x1:x2], (y1, y2, x1, x2)]
            for image, (x1, y1, x2, y2) in zip(images, boxes)]


# 데이터 생성기 (프레임 + 멜)

def datagen(frames, mels):
    """
    프레임과 멜 스펙트로그램 청크를 배치 단위로 생성
    """
    # 얼굴 박스 직접 지정 vs 검출
    if args.box[0] != -1:
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)]
                             for f in frames]
    else:
        if args.static:
            face_det_results = face_detect([frames[0]])
        else:
            face_det_results = face_detect(frames)

    img_batch, mel_batch = [], []
    frame_batch, coords_batch = [], []
    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame = frames[idx].copy()
        face, coords = face_det_results[idx]
        # 얼굴 크롭 후 96x96 리사이즈
        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame)
        coords_batch.append(coords)

        # 배치가 가득 차면 yield
        if len(img_batch) >= args.wav2lip_batch_size:
            yield prepare_batch(img_batch, mel_batch), frame_batch, coords_batch
            img_batch, mel_batch = [], []
            frame_batch, coords_batch = [], []

    # 남은 배치 처리
    if img_batch:
        yield prepare_batch(img_batch, mel_batch), frame_batch, coords_batch

# 배치 전처리 (마스크 합치기, 정규화)
def prepare_batch(imgs, mels):
    imgs = np.asarray(imgs)
    mels = np.asarray(mels)
    # 좌측 반쪽 마스크
    img_masked = imgs.copy()
    img_masked[:, :, :args.img_size//2] = 0
    # concat(masked, 원본) → (B, 96,96,6), [0,1] 정규화
    batch_x = np.concatenate((img_masked, imgs), axis=3) / 255.
    batch_m = mels[..., np.newaxis]
    return (batch_x, batch_m)


# 모델 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Inference device: {device}')

# 체크포인트 로드 (CPU/GPU 매핑)
def _load_checkpoint(cp_path):
    if device == 'cuda':
        return torch.load(cp_path)
    return torch.load(cp_path, map_location='cpu')

# Wav2Lip 모델 생성 및 가중치 로드
def load_model(cp_path):
    model = Wav2Lip()
    print(f'Checkpoint 로드: {cp_path}')
    checkpoint = _load_checkpoint(cp_path)
    state_dict = checkpoint['state_dict']
    # DataParallel 'module.' 제거
    new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state)
    model = model.to(device).eval()
    return model

# ----------------------------------------
# 6. 메인 함수
# ----------------------------------------
def main():
    # 입력 파일 검증
    if not os.path.isfile(args.face):
        raise ValueError('--face 경로가 유효하지 않습니다')

    # 정적 이미지 vs 동영상
    if args.static:
        frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        cap = cv2.VideoCapture(args.face)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 리사이즈 옵션
            if args.resize_factor > 1:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (w//args.resize_factor, h//args.resize_factor))
            # 회전 옵션
            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            # 크롭 옵션
            y1, y2, x1, x2 = args.crop
            y2 = frame.shape[0] if y2 == -1 else y2
            x2 = frame.shape[1] if x2 == -1 else x2
            frame = frame[y1:y2, x1:x2]
            frames.append(frame)
        cap.release()
    print(f'프레임 수: {len(frames)}, fps: {fps}')

    # 오디오 처리: wav가 아니면 ffmpeg로 변환
    if not args.audio.endswith('.wav'):
        temp_wav = 'temp/temp.wav'
        cmd = f'ffmpeg -y -i {args.audio} -strict -2 {temp_wav}'
        subprocess.call(cmd, shell=True)
        args.audio = temp_wav
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print(f'mel shape: {mel.shape}')

    # 멜 청크 분할 (16프레임 단위)
    mel_chunks = []
    mel_step = 16
    multiplier = 80. / fps
    i = 0
    while True:
        start = int(i * multiplier)
        if start + mel_step > mel.shape[1]:
            mel_chunks.append(mel[:, -mel_step:])
            break
        mel_chunks.append(mel[:, start:start+mel_step])
        i += 1
    print(f'mel 청크 개수: {len(mel_chunks)}')

    # 프레임과 mel 청크 길이 맞추기
    frames = frames[:len(mel_chunks)]

    # 모델 로드 및 비디오 라이터 초기화
    model = None
    out = None
    for idx, ((batch_x, batch_m), batch_frames, batch_coords) in enumerate(
            tqdm(datagen(frames, mel_chunks),
                 total=int(np.ceil(len(mel_chunks)/args.wav2lip_batch_size)))):
        if idx == 0:
            model = load_model(args.checkpoint_path)
            h, w = frames[0].shape[:2]
            os.makedirs('temp', exist_ok=True)
            out = cv2.VideoWriter('temp/result.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'),
                                  fps, (w, h))
        # 텐서 변환 및 디바이스 이동
        x = torch.FloatTensor(batch_x.transpose(0,3,1,2)).to(device)
        m = torch.FloatTensor(batch_m.transpose(0,3,1,2)).to(device)
        # 추론
        with torch.no_grad():
            pred = model(m, x)
        pred = pred.cpu().numpy().transpose(0,2,3,1) * 255.

        # 결과 프레임에 합성
        for p_frame, orig_frame, (y1, y2, x1, x2) in zip(
                pred, batch_frames, batch_coords):
            p = cv2.resize(p_frame.astype(np.uint8), (x2-x1, y2-y1))
            orig_frame[y1:y2, x1:x2] = p
            out.write(orig_frame)
    out.release()

    # 최종 ffmpeg로 비디오+오디오 합성
    final_cmd = (
        f'ffmpeg -y -i temp/result.avi -i {args.audio} '
        '-c:v libx264 -crf 16 -preset slow '
        '-c:a aac -b:a 192k -vf "scale=iw:ih" '
        f'{args.outfile}'
    )
    subprocess.call(final_cmd, shell=platform.system() != 'Windows')

if __name__ == '__main__':
    main()
