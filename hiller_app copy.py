import streamlit as st
# from ffpyplayer.player import MediaPlayer
import cv2
import openai
import queue
import threading
import os
from pathlib import Path
import numpy as np
from gtts import gTTS
import cv2
from gfpgan import GFPGANer
from streamlit.runtime.scriptrunner import add_script_run_ctx
from APP_test_utils import *
import playsound
# import winsound
import pathlib
import shutil
import base64
import time

STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
VIDEOS_PATH = (STREAMLIT_STATIC_PATH / "hiller")

if not VIDEOS_PATH.is_dir():
    VIDEOS_PATH.mkdir()


st.set_page_config(layout="wide")
HERE = Path(__file__).parent

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
source_folder = os.path.join(HERE, 'hiller')

openai.api_key = "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"
q = queue.Queue()

face_enhancer = GFPGANer(
			model_path='face_enhance.pth',
			upscale=1,
			arch='clean',
			channel_multiplier=2,
			bg_upsampler=None)

model = load_model('checkpoints/wav2lip_1.pth')
input_face = "hiller/input_face.png"

if not os.path.exists(source_folder):
      os.makedirs(source_folder)
      
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img width='250' src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html
def generate_result(input_face, input_audio, o):
  if not os.path.isfile(input_face):
    raise ValueError('--face argument must be a valid path to video/image file')
  elif input_face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    full_frames = [cv2.imread(input_face)]
    fps = 25
  else:
    video_stream = cv2.VideoCapture(input_face)
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    print('Reading video frames...')

    full_frames = []
    while 1:
      still_reading, frame = video_stream.read()
      if not still_reading:
        video_stream.release()
        break

      y1, y2, x1, x2 = 0, -1, 0, -1
      if x2 == -1: x2 = frame.shape[1]
      if y2 == -1: y2 = frame.shape[0]

      frame = frame[y1:y2, x1:x2]

      full_frames.append(frame)
  print ("Number of frames available for inference: "+str(len(full_frames)))
  
  if not input_audio.endswith('.wav'):
    print('Extracting raw audio...')
    command = 'ffmpeg -y -i {} -strict -2 {}'.format(input_audio, 'hiller/voice{}.wav'.format(o))
    subprocess.call(command, shell=True)
    input_audio = 'hiller/voice{}.wav'.format(o)
  
  wav = audio.load_wav(input_audio, 16000)
  mel = audio.melspectrogram(wav)
  print(mel.shape)
  
  if np.isnan(mel.reshape(-1)).sum() > 0:
    raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

  mel_chunks = []
  mel_idx_multiplier = 80./fps 
  i = 0
  
  while 1:
    start_idx = int(i * mel_idx_multiplier)
    if start_idx + mel_step_size > len(mel[0]):
      mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
      break
    mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
    i += 1
  print("Length of mel chunks: {}".format(len(mel_chunks)))
  
  full_frames = full_frames[:len(mel_chunks)]
  batch_size = 128
  gen = datagen(full_frames.copy(), mel_chunks)
  
  for j, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
    if j == 0:
      frame_h, frame_w = full_frames[0].shape[:-1]
      out = cv2.VideoWriter('hiller/result{}.avi'.format(o), 
                  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

    with torch.no_grad():
      pred = model(mel_batch, img_batch)

    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
    kk = 0
    for p, f, c in zip(pred, frames, coords):
      y1, y2, x1, x2 = c
      p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
      print(kk)
      # _, _, p = face_enhancer.enhance(p, has_aligned=False, only_center_face=False, paste_back=True)
      f[y1:y2, x1:x2] = p
      out.write(f)
      kk += 1

  out.release()

  command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(input_audio, 'hiller/result{}.avi'.format(o), 'hiller/result{}.mp4'.format(o))
  subprocess.call(command, shell=True)

def speak(text, i):
    tts = gTTS(text=text, lang="en")
    tts.save("hiller/voice{}.mp3".format(i))

def generate_response(prompt):
    completions = openai.Completion.create(
        engine = 'text-davinci-003',
        prompt = prompt,
        max_tokens = 1024,
        n = 1,
        temperature = 0.5,
    )
    message = completions.choices[0].text
    return message

def generate_video(q, question):
    # answer = generate_response(question)
    answer = question
    answer=answer.replace("\n","")
    answer_field.text_area("Answer from OpenAI:", value=answer, height=200, max_chars=None, key=None)
    sentences = answer.split('.')
    for i, sentence in enumerate(sentences):
        if sentence != "":
          speak(sentence, i)
          generate_result(input_face, "hiller/voice{}.mp3".format(i), i)
          shutil.copy('hiller/result{}.mp4'.format(i), VIDEOS_PATH / 'result{}.mp4'.format(i))
          q.put('hiller/result{}.mp4'.format(i))
          
def display_video(q, avatar_image):
    avatar_image.markdown(img_to_html('hiller/input_face.png'), unsafe_allow_html=True)
    while True:
      # if q.empty():
        # video_html = """
        #     <video controls width="250" autoplay="true" muted="true" loop="true>
        #     <source 
        #         src="hiller/default.mp4" 
        #         type="video/mp4" />
        #     </video>
        # """
        # avatar_image.markdown(video_html, unsafe_allow_html=True)
        
        # avatar_image.image(input_face)
      if not q.empty():
        current_video = q.get()
        video_html = """
            <video width="250" autoplay="true" muted="true">
            <source 
                src={}
                type="video/mp4" />
            </video>
        """.format(current_video)
        print('current video: ', current_video)
        avatar_image.markdown(video_html, unsafe_allow_html=True)
        # time.sleep(0.1)
        playsound.playsound("hiller/voice{}.mp3".format(current_video.split("/")[1].split(".")[0][-1]), True)
        
        # cap = cv2.VideoCapture(current_video)
        # # player = MediaPlayer("hiller/voice0.mp3")
        # playsound.playsound("hiller/voice{}.mp3".format(current_video.split("/")[1].split(".")[0][-1]), False)
        # # winsound.PlaySound("hiller/voice0.mp3", winsound.SND_ASYNC)
        # if (cap.isOpened()== False): 
        #   print("Error opening video stream or file")
        # while(cap.isOpened()):
        #   ret, frame = cap.read()
        #   if ret == True:    
        #       avatar_image.image(frame,channels='BGR')              
        #   else: 
        #       print('End')
        #       cap.release()
        #       break
        # # player.close_player()
        # # winsound.PlaySound(None, winsound.SND_PURGE)
        os.remove("hiller/voice{}.mp3".format(current_video.split("/")[1].split(".")[0][-1]))

if __name__ == "__main__":
      
  col1, col2 = st.columns(2)
  with col2:
      st.header("Real Time Video Avatar")
      avatar_image = st.empty()
      
      t2 = threading.Thread(target=display_video, args=(q, avatar_image))
      add_script_run_ctx(t2)
      t2.start()
  with col1:
      st.header("Chatbot Interface")
      question = st.text_input("Please enter your question here:")
      answer_field = st.empty()
      if question:        
          t1 = threading.Thread(target=generate_video, args=(q,question))
          add_script_run_ctx(t1)
          t1.start()
  t2.join()