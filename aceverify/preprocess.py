import argparse
import subprocess
import logging
import shutil
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
import zipfile
import h5py
import os
import numpy as np
import json
from scipy.io import wavfile
from typing import Union
import torch

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMP_DIR = PROJECT_ROOT / 'temp'


def configure_logging(log_level: str = "INFO"):
  numeric_level = getattr(logging, log_level.upper(), logging.INFO)
  logging.basicConfig(
      level=numeric_level,
      format='%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s'
  )


# def resolve_ffmpeg_executable(ffmpeg_bin: str | None = None) -> str:
def resolve_ffmpeg_executable(ffmpeg_bin: Union[str, None] = None) -> str:
  if ffmpeg_bin:
    return ffmpeg_bin

  env_ffmpeg_bin = os.environ.get('FFMPEG_BIN')
  if env_ffmpeg_bin:
    return env_ffmpeg_bin

  resolved = shutil.which('ffmpeg')
  if resolved:
    return resolved

  raise RuntimeError(
      'ffmpeg executable not found. Install ffmpeg and ensure it is in PATH, '
      'or pass --ffmpeg-bin /absolute/path/to/ffmpeg, or set FFMPEG_BIN.'
  )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)

def process_vid(file_name, subfolder, temp_dir: Path, ffmpeg_bin: str):
  video_path = temp_dir / subfolder / f'{file_name}.mp4'
  frame_out_pattern = str(temp_dir / f'{file_name}_%02d.jpg')
  audio_out_path = str(temp_dir / f'{file_name}_audio.wav')
  logger.debug('Starting video processing for %s', video_path)
  try:
    subprocess.run([ffmpeg_bin, '-loglevel', 'error', '-ss', '00:00:05', '-i', str(video_path), '-frames:v', '16', '-q:v', '2', frame_out_pattern], check=True)
    subprocess.run([ffmpeg_bin, '-loglevel', 'error', '-ss', '00:00:05', '-i', str(video_path), '-vn', '-t', '0.5', '-acodec', 'pcm_s16le', audio_out_path], check=True)
  except subprocess.CalledProcessError:
    logger.exception('ffmpeg command failed for video %s', video_path)
    return False
  except (FileNotFoundError, NotADirectoryError, OSError):
    logger.exception('ffmpeg executable is invalid/unavailable: %s', ffmpeg_bin)
    return False
  except Exception:
    logger.exception('Unexpected error while extracting frames/audio for %s', video_path)
    return False

  face_box = None
  for i in range(1,17):
    image_path = temp_dir / f'{file_name}_{i:02d}.jpg'
    try:
      with Image.open(image_path) as image:
        face_boxes, _ = mtcnn.detect(image)
        if face_boxes is not None:
          face_box = face_boxes[0].astype(int)
          face_box[0] -= 80
          face_box[1] -= 50
          face_box[2] += 80
          face_box[3] += 50
          break
    except Exception:
      logger.exception('Failed during face detection for frame %s', image_path)
      return False

  if (face_box is None):
    logger.warning('No face detected in extracted frames for %s', file_name)
    return False

  for i in range(1,17):
    image_path = temp_dir / f'{file_name}_{i:02d}.jpg'
    save_path = temp_dir / f'{file_name}_processed_{i:02d}.jpg'
    
    try:
      with Image.open(image_path).convert('RGB') as image:
        image_cropped = image.crop(face_box).resize((224,224))
        image_cropped.save(save_path)
    except Exception:
      logger.exception('Failed to crop/save processed frame %s', image_path)
      return False

  return True

def save_vid_to_h5(file_name, label, h5_path, temp_dir: Path):
  logger.debug('Saving processed sample %s to %s with label=%s', file_name, h5_path, label)
  frames_16 = []

  for i in range(1,17):
    image_path = temp_dir / f'{file_name}_processed_{i:02d}.jpg'
    try:
      with Image.open(image_path) as image:
        image_nparray = np.array(image)
        frames_16.append(image_nparray)
    except Exception:
      logger.exception('Failed to load processed image %s', image_path)
      raise
  vid_data_final = np.stack(frames_16)

  audio_path = temp_dir / f'{file_name}_audio.wav'
  try:
    fs, audio_data = wavfile.read(audio_path)
    logger.debug('Read audio from %s at %s Hz', audio_path, fs)
  except Exception:
    logger.exception('Failed to read extracted audio %s', audio_path)
    raise

  try:
    with h5py.File(h5_path, 'a') as f:
      if file_name in f:
        del f[file_name]
      g = f.create_group(file_name)
      g.create_dataset('video', data=vid_data_final, compression='gzip')
      g.create_dataset('audio', data=audio_data, compression='gzip')
      g.attrs['label'] = label
  except Exception:
    logger.exception('Failed to write sample %s to h5 file %s', file_name, h5_path)
    raise

def delete_preprocessed_files(file_name, subfolder, temp_dir: Path):
  raw_vid_path = temp_dir / subfolder / f'{file_name}.mp4'
  if os.path.exists(raw_vid_path):
    try:
      os.remove(raw_vid_path)
    except Exception:
      logger.exception('Failed to remove raw video %s', raw_vid_path)
  for i in range(1,17):
    ith_jpeg_path = temp_dir / f'{file_name}_{i:02d}.jpg'
    if os.path.exists(ith_jpeg_path):
      try:
        os.remove(ith_jpeg_path)
      except Exception:
        logger.exception('Failed to remove extracted frame %s', ith_jpeg_path)
    ith_processed_jpeg_path = temp_dir / f'{file_name}_processed_{i:02d}.jpg'
    if os.path.exists(ith_processed_jpeg_path):
      try:
        os.remove(ith_processed_jpeg_path)
      except Exception:
        logger.exception('Failed to remove processed frame %s', ith_processed_jpeg_path)
  audio_path = temp_dir / f'{file_name}_audio.wav'
  if os.path.exists(audio_path):
    try:
      os.remove(audio_path)
    except Exception:
      logger.exception('Failed to remove temporary audio %s', audio_path)

def preprocess_dataset(zip_file_path, subfolder, h5_path, temp_dir: str | Path | None = None, ffmpeg_bin: str | None = None):
  if temp_dir is None:
    temp_dir_path = DEFAULT_TEMP_DIR
  else:
    temp_dir_path = Path(temp_dir).expanduser().resolve()
  temp_dir_path.mkdir(parents=True, exist_ok=True)

  ffmpeg_executable = resolve_ffmpeg_executable(ffmpeg_bin)

  logger.info('Starting preprocessing: zip=%s, subfolder=%s, output=%s', zip_file_path, subfolder, h5_path)
  logger.info('Using temp directory: %s', temp_dir_path)
  logger.info('Using ffmpeg executable: %s', ffmpeg_executable)
  with h5py.File(h5_path, 'w') as f:
    logger.info('Created h5 file at %s', h5_path)

  with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_file_names = zip_ref.namelist()

    label_sheet = [f for f in zip_file_names if f.endswith('.json')]
    if not label_sheet:
      raise ValueError(f'No label json file found in zip: {zip_file_path}')
    label_sheet = label_sheet[0]
    with zip_ref.open(label_sheet) as f:
      json_data = json.loads(f.read().decode('utf-8'))
  
    files_to_process = [f for f in zip_file_names if f.endswith('.mp4')]
    logger.info('Found %d videos to process', len(files_to_process))
    
    # extract each video in zip and process
    for file in files_to_process:
      try:
        zip_ref.extract(file, path=temp_dir_path)
        file_name_with_ext = os.path.basename(file)
        file_name = file_name_with_ext.removesuffix('.mp4')
        label_string = json_data[file_name_with_ext]['label']
        if (label_string == 'FAKE'):
          label = 1
        else:
          label = 0

        logger.info('Processing video=%s label=%s (%s)', file_name, label_string, label)
        processed = process_vid(file_name, subfolder, temp_dir_path, ffmpeg_executable)
        if processed:
          save_vid_to_h5(file_name, label, h5_path, temp_dir_path)
          delete_preprocessed_files(file_name, subfolder, temp_dir_path)
          logger.info('Successfully processed and saved %s', file_name)
        else:
          logger.warning('Skipped video due to preprocessing failure: %s', file_name)
      except Exception:
        logger.exception('Unhandled error while processing zip entry %s', file)

def main():
  parser = argparse.ArgumentParser(description='Preprocess deepfake videos into HDF5 format.')
  parser.add_argument('zip_file', help='Path to the zip file containing raw video data (e.g. dfdc_train_part_00.zip)')
  parser.add_argument('subfolder', help='Subfolder name within the zip file to use as the temp extraction prefix (e.g. dfdc_train_part_0)')
  parser.add_argument('--output', '-o', default='processed_data.h5',
                      help='Path to the output HDF5 file (default: processed_data.h5)')
  parser.add_argument('--temp-dir', default=str(DEFAULT_TEMP_DIR),
                      help='Directory for temporary extracted/processed files (default: <repo>/temp)')
  parser.add_argument('--ffmpeg-bin', default=None,
                      help='Path to ffmpeg executable (default: resolve from FFMPEG_BIN or PATH)')
  parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging verbosity (default: INFO)')
  args = parser.parse_args()

  configure_logging(args.log_level)

  zip_file_path = args.zip_file
  subfolder = args.subfolder
  h5_path = args.output

  try:
    preprocess_dataset(zip_file_path, subfolder, h5_path, temp_dir=args.temp_dir, ffmpeg_bin=args.ffmpeg_bin)
  except Exception:
    logger.exception('Fatal error in preprocessing pipeline')
    raise

if __name__ == "__main__":
  main()