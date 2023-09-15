from pytube import YouTube
from moviepy.editor import *
import csv
import os

# Initialize variables
total_videos = 0
successful_downloads = 0

# Your CSV file path and download folder
csv_file_path = 'test.csv'
download_folder = 'AudioCaps/'

# Read CSV and download YouTube videos
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader, None)  # Skip the header
    
    for row in csv_reader:
        total_videos += 1
        audiocap_id, youtube_id, start_time, caption = row
        
        try:
            yt = YouTube(f'https://www.youtube.com/watch?v={youtube_id}')
            # Download the highest quality audio
            ys = yt.streams.filter(only_audio=True).first()
            audio_file_path = ys.download(output_path=download_folder)
            
            # Convert to WAV using moviepy
            audio = AudioFileClip(audio_file_path)
            audio.write_audiofile(download_folder + youtube_id + ".wav")

            # Remove the original audio file (.webm)
            os.remove(audio_file_path)
            
            print(f"Downloaded and converted: {youtube_id}")
            successful_downloads += 1
        except Exception as e:
            print(f"An error occurred for video {youtube_id}: {e}")

# Calculate the percentage of successful downloads
if total_videos > 0:
    success_percentage = (successful_downloads / total_videos) * 100
else:
    success_percentage = 0

print(f"Total Videos: {total_videos}")
print(f"Successful Downloads: {successful_downloads}")
print(f"Success Percentage: {success_percentage}%")
