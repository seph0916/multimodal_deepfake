import os
import shutil


# def traverse_directory(directory):
#     # 디렉토리 안의 파일 및 하위 디렉토리 목록을 얻습니다.
#     for root, dirs, files in os.walk(directory):
#         dirs=sorted(dirs)
#         # 하위 디렉토리에 대해 반복합니다.
#         for dir_name in dirs:
#             subdir_path = os.path.join(root, dir_name)
#             subdir_path = os.path.join(subdir_path,dir_name)
            


# # 주어진 디렉토리 경로를 사용하여 함수를 호출합니다.
# directory_path = "/home/minkyo/data_avceleb/data_fakeav/val/real"
# output_folder = "/home/josephlee/multimodal/audio/val/real"
# traverse_directory(directory_path)






# 이건 뭐지이??
# def copy_images(source_dir, dest_dir):
#     # 소스 디렉토리에 있는 모든 파일 목록 가져오기
#     files = os.listdir(source_dir)
#     # 이미지 파일만 선택
#     files=sorted(files)
#     image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
#     # 대상 디렉토리가 없다면 생성
#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)
    
#     # 이미지 파일을 대상 디렉토리로 복사
#     for image_file in image_files:
#         source_path = os.path.join(source_dir, image_file)
#         dest_path = os.path.join(dest_dir, image_file)
#         shutil.copyfile(source_path, dest_path)
#         print(f"이미지 '{image_file}'를 복사했습니다.")

# # 소스 디렉토리와 대상 디렉토리 경로 설정
# source_directory = '/home/josephlee/multimodal/image/test/fake'
# destination_directory = '/home/josephlee/multimodal/image/gogo'


# copy_images(source_directory, destination_directory)




# train, val test에 wav갯수에 맞춘 이미지 갯수를 비레로 input set으로 보내기
import os
from collections import defaultdict


###
# directory = "/home/josephlee/multimodal/audio/train/fake"
# wav_files_per_id = count_wav_files_per_id(directory)
# import os
# import shutil
# def copy_images_for_wavs(wav_files_per_id, source_directory, destination_directory, num_images_per_wav):
#     # 각 ID에 해당하는 WAV 파일을 순회하면서 이미지 파일을 복사합니다.
#     for file_id, wav_count in wav_files_per_id.items():
#         # 해당 ID의 디렉토리 경로를 구합니다.
#         id_directory = os.path.join(source_directory, f"{file_id}_fake")
#         # 해당 ID의 디렉토리에 있는 모든 WAV 파일에 대해 이미지 파일을 복사합니다.
#         for i in range(0,wav_count * num_images_per_wav):
#             image_index = i + 1
#             source_image_path = os.path.join(id_directory, f"{file_id}_fake_{image_index}.jpg")
#             destination_image_path = os.path.join(destination_directory, os.path.basename(source_image_path))
#             shutil.copyfile(source_image_path, destination_image_path)
#             print(f"Copied {source_image_path} to {destination_image_path}")

# num_images_per_wav=25
# source_directory = "/home/josephlee/multimodal/datafold/train/fake"
# destination_directory = "/home/josephlee/multimodal/image/train/fake"
# copy_images_for_wavs(wav_files_per_id, source_directory, destination_directory, num_images_per_wav)
###

def count_wav_files_per_id(directory):
    # 디렉토리 내의 모든 파일과 하위 디렉토리의 파일을 순회하며 WAV 파일을 찾습니다.
    wav_files_per_id = defaultdict(int)
    for root, dirs, files in os.walk(directory):
        files=sorted(files)
        for file in files:
            if file.endswith(".wav"):
                # 파일 이름에서 "id" 뒤의 숫자를 추출하여 해당 숫자의 파일 개수를 증가시킵니다.
                file_id = file.split("_")[0]
                wav_files_per_id[file_id] += 1
    return wav_files_per_id



### real에 대한 부분
num_images_per_wav = 25  # 1개의 WAV 파일당 복사할 이미지 수
directory = "/home/josephlee/multimodal/audio/train/real"
wav_files_per_id = count_wav_files_per_id(directory)

        
import os
import shutil

def copy_images_for_wavs(wav_files_per_id, source_directory, destination_directory, num_images_per_wav):
    # 각 ID에 해당하는 WAV 파일을 순회하면서 이미지 파일을 복사합니다.
    for file_id, wav_count in wav_files_per_id.items():
        # 해당 ID의 디렉토리 경로를 구합니다.
        id_directory = os.path.join(source_directory, f"{file_id}")
        # 해당 ID의 디렉토리에 있는 모든 WAV 파일에 대해 이미지 파일을 복사합니다.
        for i in range(0,wav_count * num_images_per_wav):
            image_index = i + 1
            source_image_path = os.path.join(id_directory, f"{file_id}_{image_index}.jpg")
            destination_image_path = os.path.join(destination_directory, os.path.basename(source_image_path))
            shutil.copyfile(source_image_path, destination_image_path)
            print(f"Copied {source_image_path} to {destination_image_path}")


source_directory = "/home/minkyo/data_avceleb/data_fakeav/train/real"
destination_directory = "/home/josephlee/multimodal/image/train/real"
copy_images_for_wavs(wav_files_per_id, source_directory, destination_directory, num_images_per_wav)
