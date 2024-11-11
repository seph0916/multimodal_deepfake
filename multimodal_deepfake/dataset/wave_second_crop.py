from pydub import AudioSegment

def split_wav_by_seconds(input_file, output_folder,dir_name):
    sound = AudioSegment.from_wav(input_file)
    duration_in_sec = len(sound) / 1000  # convert to seconds

    for i in range(0, int(duration_in_sec)):
        start_time = i * 1000  # convert to milliseconds
        end_time = (i + 1) * 1000  # convert to milliseconds
        segment = sound[start_time:end_time]
        segment.export(f"{output_folder}"+"/"+dir_name+f"_{i + 1}.wav", format="wav")

# 스타트타임 엔드타임 간격 1/fps
import os

def traverse_directory(directory):
    # 디렉토리 안의 파일 및 하위 디렉토리 목록을 얻습니다.
    for root, dirs, files in os.walk(directory):
        dirs=sorted(dirs)
        # 하위 디렉토리에 대해 반복합니다.
        for dir_name in dirs:
            subdir_path = os.path.join(root, dir_name)
            subdir_path = os.path.join(subdir_path,dir_name+'.wav')
            split_wav_by_seconds(subdir_path,output_folder,dir_name)


# 주어진 디렉토리 경로를 사용하여 함수를 호출합니다.
directory_path = "/home/minkyo/data_avceleb/data_fakeav/train/fake"
output_folder = "/home/josephlee/multimodal/audio/train/fake"
traverse_directory(directory_path)
