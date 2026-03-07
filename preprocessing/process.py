import argparse
import ffmpeg
import subprocess
from facenet_pytorch import MTCNN
from PIL import Image
import zipfile
import h5py
import os
import numpy as np
import json
from scipy.io import wavfile
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)

def process_vid(file_name, subfolder):
  try:
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-ss', '00:00:05', '-i', f'../temp/{subfolder}/{file_name}.mp4', '-frames:v', '16', '-q:v', '2', f'../temp/{file_name}_%02d.jpg'], check=True)
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-ss', '00:00:05', '-i', f'../temp/{subfolder}/{file_name}.mp4', '-vn', '-t', '0.5', '-acodec', 'pcm_s16le', f'../temp/{file_name}_audio.wav'], check=True)
  except subprocess.CalledProcessError:
    return False
  except FileNotFoundError:
    return False

  face_box = None
  for i in range(1,17):
    image_path = f'../temp/{file_name}_{i:02d}.jpg'
    with Image.open(image_path) as image:
      face_boxes, _ = mtcnn.detect(image)
      if face_boxes is not None:
        face_box = face_boxes[0].astype(int)
        face_box[0] -= 80
        face_box[1] -= 50
        face_box[2] += 80
        face_box[3] += 50
        break

  if (face_box is None):
    return False

  for i in range(1,17):
    image_path = f'../temp/{file_name}_{i:02d}.jpg'
    save_path = f'../temp/{file_name}_processed_{i:02d}.jpg'
    
    with Image.open(image_path).convert('RGB') as image:
      image_cropped = image.crop(face_box).resize((224,224))
      image_cropped.save(save_path)

  return True

def save_vid_to_h5(file_name, label, h5_path):
  frames_16 = []

  for i in range(1,17):
    image_path = f'../temp/{file_name}_processed_{i:02d}.jpg'
    with Image.open(image_path) as image:
      image_nparray = np.array(image)
      frames_16.append(image_nparray)
  vid_data_final = np.stack(frames_16)

  audio_path = f'../temp/{file_name}_audio.wav'
  fs, audio_data = wavfile.read(audio_path)

  with h5py.File(h5_path, 'a') as f:
    if file_name in f: 
      del f[file_name]
    g = f.create_group(file_name)
    d1 = g.create_dataset('video', data=vid_data_final, compression='gzip')
    d2 = g.create_dataset('audio', data=audio_data, compression='gzip')
    g.attrs['label'] = label

def delete_preprocessed_files(file_name, subfolder):
  raw_vid_path = f'../temp/{subfolder}/{file_name}.mp4'
  if os.path.exists(raw_vid_path):
    os.remove(raw_vid_path)
  for i in range(1,17):
    ith_jpeg_path = f'../temp/{file_name}_{i:02d}.jpg'
    if os.path.exists(ith_jpeg_path):
      os.remove(ith_jpeg_path)
    ith_processed_jpeg_path = f'../temp/{file_name}_processed_{i:02d}.jpg'
    if os.path.exists(ith_processed_jpeg_path):
      os.remove(ith_processed_jpeg_path)
  audio_path = f'../temp/{file_name}_audio.wav'
  if os.path.exists(audio_path):
    os.remove(audio_path)


def main():
  parser = argparse.ArgumentParser(description='Preprocess deepfake detection videos into HDF5 format.')
  parser.add_argument('zip_file', help='Path to the zip file containing raw video data (e.g. dfdc_train_part_00.zip)')
  parser.add_argument('subfolder', help='Subfolder name within the zip file to use as the temp extraction prefix (e.g. dfdc_train_part_0)')
  parser.add_argument('--output', '-o', default='processed_data.h5',
                      help='Path to the output HDF5 file (default: processed_data.h5)')
  args = parser.parse_args()

  zip_file_path = args.zip_file
  subfolder = args.subfolder
  h5_path = args.output

  with h5py.File(h5_path, 'w') as f:
    print('created h5 file!')

  with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_file_names = zip_ref.namelist()

    label_sheet = [f for f in zip_file_names if f.endswith('.json')]
    label_sheet = label_sheet[0]
    with zip_ref.open(label_sheet) as f:
      json_data = json.loads(f.read().decode('utf-8'))
  
    files_to_process = [f for f in zip_file_names if f.endswith('.mp4')]
    
    # extract each video in zip and process
    for file in files_to_process:
      zip_ref.extract(file, path='../temp')
      file_name_with_ext = os.path.basename(file)
      file_name = file_name_with_ext.removesuffix('.mp4')
      label_string = json_data[file_name_with_ext]['label']
      if (label_string == 'FAKE'):
        label = 1 
      else:
        label = 0

      print(f'processing {label_string} ({label}) video {file_name}........')
      processed = process_vid(file_name, subfolder)
      if processed:
        save_vid_to_h5(file_name, label, h5_path)
        delete_preprocessed_files(file_name, subfolder)
        print(f'{file_name} processed and saved!')
      else: 
        print(f'could not process {file_name}')


if __name__ == "__main__":
  main()