from openai import OpenAI
import os
import cv2
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
import time
import base64

# Load environment variables from .env file
load_dotenv()

MODEL="gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VIDEO_PATH = "video.mp4"

def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = 0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    # Extract audio from video
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path

# Extract 1 frame per second. Adjust `seconds_per_frame` to change sampling rate
base64Frames, audio_path = process_video(VIDEO_PATH, seconds_per_frame=1)

transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=open(audio_path, "rb"),
)

# Info Gatherer
response = client.chat.completions.create(
    model=MODEL,
    messages=[
      {"role":"system","content":"""You are generating summary from a video. Give a detailed summary of the video based on the provided frames of the video and its transcript."""},
      {"role":"user","content":[
          "These are the frames from the video.",
          *map(lambda x:{"type":"image_url",
                         "image_url":{"url":f'data:image/jpg;base64,{x}',"detail":"low"}}, base64Frames),
          {"type":"text","text":f"The audio transcription is: {transcription.text}"}
        ],
      }
    ],
    temperature=0,
)
print(response.choices[0].message.content)
