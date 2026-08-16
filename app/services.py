import ffmpeg
import subprocess
from facenet_pytorch import MTCNN
from PIL import Image
import h5py
import os
import numpy as np
from scipy.io import wavfile
import torch
import streamlit as st
import gdown
import cv2
import matplotlib.pyplot as plt
from aceverify.model import load_from_checkpoint
from aceverify.dataset import ACEDataset
from src.attention_map import predict_video, attention_map

def process_vid(file_name, folder_path):
  try:
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-ss', '00:00:05', '-i', f'{folder_path}/{file_name}.mp4', '-frames:v', '16', '-q:v', '2', f'{folder_path}/{file_name}_%02d.jpg'], check=True)
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-ss', '00:00:05', '-i', f'{folder_path}/{file_name}.mp4', '-vn', '-t', '0.5', '-acodec', 'pcm_s16le', f'{folder_path}/{file_name}_audio.wav'], check=True)
  except subprocess.CalledProcessError:
    return False
  except FileNotFoundError:
    return False
  
  mtcnn = MTCNN(keep_all=False)

  face_box = None
  for i in range(1,17):
    image_path = f'{folder_path}/{file_name}_{i:02d}.jpg'
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
    image_path = f'{folder_path}/{file_name}_{i:02d}.jpg'
    save_path = f'{folder_path}/{file_name}_processed_{i:02d}.jpg'
    
    with Image.open(image_path).convert('RGB') as image:
      image_cropped = image.crop(face_box).resize((224,224))
      image_cropped.save(save_path)

  return True

def save_vid_to_h5(file_name, folder_path):
  h5_path = f'{folder_path}/{file_name}.h5'
  
  frames_16 = []
  for i in range(1,17):
    image_path = f'{folder_path}/{file_name}_processed_{i:02d}.jpg'
    with Image.open(image_path) as image:
      image_nparray = np.array(image)
      frames_16.append(image_nparray)
  vid_data_final = np.stack(frames_16)

  audio_path = f'{folder_path}/{file_name}_audio.wav'
  fs, audio_data = wavfile.read(audio_path)

  with h5py.File(h5_path, 'a') as f:
    g = f.create_group(file_name)
    d1 = g.create_dataset('video', data=vid_data_final, compression='gzip')
    d2 = g.create_dataset('audio', data=audio_data, compression='gzip')
    g.attrs['label'] = 2

@st.cache_resource
def load_model():
  model_file_name = 'aceverify_final.pth'
  model_path = f'app/{model_file_name}'
  url = 'https://drive.google.com/file/d/1d3ln2laSfmXkKyXHZ1YhK_gb33nonaPO/view?usp=sharing'

  if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False, fuzzy=True)
  
  # The published checkpoint may predate the multi-domain architecture, so pick
  # the class that matches the weights rather than assuming the current one.
  model, architecture = load_from_checkpoint(model_path, map_location='cpu')
  print(f'Loaded {architecture} architecture from {model_path}')
  model.eval()

  return model

def predict(file_name, folder_path, model):
  video_data = ACEDataset(h5_path=f'{folder_path}/{file_name}.h5', is_training=False)
  video, spec, _ = video_data[0]
  label, confidence = predict_video(model, 'cpu', video, spec)
  return label, confidence, video, spec

def return_attention_map(model, video_tensor, frame_idx=8):
  model.eval()
  attention_outputs = []

  def hook_fn(module, input, output):
      attention_outputs.append(output)

  try:
      target_layer = model.video_model.blocks[-1].attn.qkv
      handle = target_layer.register_forward_hook(hook_fn)
  except AttributeError:
      print("Error")
      return

  try:
      with torch.no_grad():
          input_video = video_tensor.permute(1, 0, 2, 3).to('cpu')
          _ = model.video_model(input_video)

      if not attention_outputs:
          print("Hook failed to capture data.")
          return

      qkv_out = attention_outputs[0]
      B, N, C3 = qkv_out.shape
      num_heads = 12  
      head_dim = (C3 // 3) // num_heads
  
      qkv = qkv_out.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
      q, k, v = qkv[0], qkv[1], qkv[2]

      attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
      attn = attn.softmax(dim=-1)
      attn_mean = attn.mean(dim=1)
      cls_attn = attn_mean[:, 0, 1:] 
      
      mask = cls_attn[frame_idx].reshape(14, 14).cpu().numpy()
      img_np = video_tensor[:, frame_idx, :, :].permute(1, 2, 0).cpu().numpy()
      mask_resized = cv2.resize(mask, (224, 224))
      mask_norm = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min() + 1e-8)

      # Apply Heatmap
      heatmap = cv2.applyColorMap(np.uint8(255 * mask_norm), cv2.COLORMAP_JET)
      heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
      overlay = (0.4 * heatmap + 0.6 * img_np)
      overlay = np.clip(overlay, 0, 1)

      fig, ax = plt.subplots()
      ax.imshow(overlay)
      ax.set_title('Video Frame Attention Map')
      ax.axis('off')
      st.pyplot(fig)

  finally:
      handle.remove()

def return_spec_vis(spec):
  fig, ax = plt.subplots()
  ax.imshow(spec.squeeze().numpy(), aspect='auto', origin='lower')
  ax.set_title('Mel-Spectrogram')
  ax.axis('off')
  st.pyplot(fig)

 

def delete_video_files(file_name, folder_path):
  raw_vid_path = f'{folder_path}/{file_name}.mp4'
  if os.path.exists(raw_vid_path):
    os.remove(raw_vid_path)
  for i in range(1,17):
    ith_jpeg_path = f'{folder_path}/{file_name}_{i:02d}.jpg'
    if os.path.exists(ith_jpeg_path):
      os.remove(ith_jpeg_path)
    ith_processed_jpeg_path = f'{folder_path}/{file_name}_processed_{i:02d}.jpg'
    if os.path.exists(ith_processed_jpeg_path):
      os.remove(ith_processed_jpeg_path)
  audio_path = f'{folder_path}/{file_name}_audio.wav'
  if os.path.exists(audio_path):
    os.remove(audio_path)