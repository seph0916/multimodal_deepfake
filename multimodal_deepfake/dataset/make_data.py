# audio image data loader
import os




# path='/home/minkyo/data_avceleb/data_fakeav/test'

# # 현재 디렉토리의 하위 디렉토리 목록을 가져옴
# directories = [d for d in os.listdir(path) if os.path.isdir(path)]

# # 출력
# print("하위 디렉토리 목록:")
# for directory in directories:
#     label_dir=os.path.join(path,directory)
#     for data in os.listdir(label_dir):
#         print(data)



# from moviepy.editor import VideoFileClip

# def get_video_duration(file_path):
#     try:
#         clip = VideoFileClip(file_path)
#         duration = clip.duration
#         clip.close()
#         return duration
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return None

# # 사용 예시

# duration = get_video_duration(file_path)
# if duration:
#     print(f"영상의 길이는 {duration} 초입니다.")
# else:
#     print("영상의 길이를 가져올 수 없습니다.")



# from moviepy.editor import VideoFileClip,concatenate_videoclips

# def make_video_8_seconds(input_video_path, output_video_path):
#     # 원본 영상 로드
#     clip = VideoFileClip(input_video_path)
    
#     # 영상의 길이 계산
#     duration = clip.duration
    
#     # 만약 영상의 길이가 8초보다 작다면, 앞 부분을 반복하여 채움
#     if duration < 8:
#         while clip.duration < 8:
#             clip = concatenate_videoclips([clip, clip.subclip(0, min(clip.duration, 8 - clip.duration))])
    
#     # 8초를 넘어가면, 뒷 부분을 잘라냄
#     if duration > 8:
#         clip = clip.subclip(0, 8)
    
#     # 새로운 영상을 파일로 저장
#     clip.write_videofile(output_video_path)

# # 예시: 입력과 출력 경로 설정
# input_video_path = "/home/josephlee/multimodal/aa/id00018_fake.mp4"
# output_video_path = "/home/josephlee/multimodal/aa/id00018_change.mp4"

# # 함수 호출
# make_video_8_seconds(input_video_path, output_video_path)
# file_path = "/home/josephlee/multimodal/aa/id00018_change.mp4"  # 파일 경로를 올바르게 지정해주세요.
# clip=VideoFileClip(file_path)
# clip.duration

import librosa
import cv2
import numpy as np 

def load_audio(filename, target_duration):
    # WAV 파일 로드
    y, sr = librosa.load(filename)
    
    # 현재 오디오의 길이
    duration = librosa.get_duration(y=y, sr=sr)
    
    if duration < target_duration:
        # 부족한 부분을 0으로 채워서 길이를 맞춰줍니다.
        y = librosa.util.fix_length(y, int(target_duration * sr))
    
    return y, sr

import os
import cv2

def load_images_from_directory(directory_path, target_duration, sr):
    images = []
    
    # 디렉토리 내의 모든 파일을 가져옵니다.
    for filename in os.listdir(directory_path):
        # 파일의 절대 경로
        file_path = os.path.join(directory_path, filename)
        
        # 파일이 이미지인지 확인합니다.
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 이미지 파일 로드
            img = cv2.imread(file_path)
            
            # 이미지의 총 프레임 수
            total_frames = img.shape[1]
            
            # 원하는 시간까지의 프레임 수 계산
            target_frames = int(target_duration * sr)
            
            if total_frames >= target_frames:
                # 이미지를 잘라냅니다.
                img = img[:, :target_frames, :]
            else:
                # 부족한 부분은 이미지의 마지막 프레임을 반복해서 채워줍니다.
                repeat_frames = target_frames - total_frames
                repeated_img = np.tile(img[:, -1:, :], (1, repeat_frames, 1))
                img = np.concatenate((img, repeated_img), axis=1)
            
            images.append(img)
    
    return images


# 예시 사용법
target_duration = 8  # 기준 음악 길이

# WAV 파일과 이미지 파일 경로
wav_filename = '/home/minkyo/data_avceleb/data_fakeav/test/fake/id00043_fake/id00043_fake.wav'
directory_path = '/home/minkyo/data_avceleb/data_fakeav/test/fake/id00043_fake'
# 음악 데이터 로드
audio_data, sample_rate = load_audio(wav_filename, target_duration)


# 이미지 데이터 로드
image_data = load_images_from_directory(directory_path, target_duration, sample_rate)


