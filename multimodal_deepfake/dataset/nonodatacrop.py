import os
import cv2

# 비디오가 위치한 디렉토리
video_dir = '/home/josephlee/multimodal/image/train/fake/'

# 이미지를 저장할 디렉토리
image_dir = '/home/josephlee/multimodal/image/train/fake/'

# 비디오 디렉토리 내의 모든 하위 디렉토리 가져오기
sub_dirs = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]

# 하위 디렉토리 확인 및 생성
for sub_dir in sub_dirs:
    sub_image_dir = os.path.join(image_dir, sub_dir)
    if not os.path.exists(sub_image_dir):
        os.makedirs(sub_image_dir)

    # 해당 하위 디렉토리 내의 비디오 목록 가져오기
    videos = [v for v in os.listdir(os.path.join(video_dir, sub_dir)) if v.endswith('.mp4')]

    # 비디오를 프레임 단위로 처리하여 이미지로 저장
    for video in videos:
        video_path = os.path.join(video_dir, sub_dir, video)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 초당 10프레임만 저장
            if frame_count % 10 == 0:
                image_path = os.path.join(sub_image_dir, f'{video[:-4]}_{frame_count//10}.jpg')
                cv2.imwrite(image_path, frame)
            frame_count += 1
        cap.release()

print("작업 완료!") 