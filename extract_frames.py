# pip install pandas
# pip install openpyxl
# conda install -c conda-forge opencv

import pandas as pd
import cv2
import os
import subprocess

def extract_multiple_frames(video_path, target_seconds, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = {int(s * fps): s for s in target_seconds if pd.notna(s)}
    
    current_frame = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while current_frame < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in target_frames:
            second = target_frames[current_frame]
            output_path = os.path.join(output_folder, f"{video_name}_frame_at_{second}s.png")
            cv2.imwrite(output_path, frame)
            print(f"Saved frame at {second}s → {output_path}")

        current_frame += 1

    cap.release()
    
def extract_frames_at_times_FFmpeg(video_path, timestamps, output_folder):
    for t in timestamps:
        if pd.isna(t):
            continue
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{output_folder}/{video_name}_frame_at_{t}s.png"
        command = [
            'ffmpeg',
            '-ss', str(t),
            '-i', video_path,
            '-frames:v', '1',
            '-q:v', '2',
            output_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Extracted frame at {t}s → {output_path}")

# Process each video and its corresponding target seconds
def process_videos_from_excel(excel_path, output_folder, skip):
    # Read the Excel file
    df = pd.read_excel(excel_path, header=None)  # Read without headers
    
    # Make sure there's at least one column (video path column)
    if df.empty or len(df.columns) < 2:
        print("Error: Excel sheet doesn't contain video paths or seconds.")
        return
    
    # Loop over each row (i.e., each video)
    for index, row in df.iterrows():
        if index < skip:
            continue
            
        video_path = row[0]  # Read video path from the first column
        print(f"Processing video: {video_path}")
        
        # Ensure the video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist.")
            continue
            
        # extract_multiple_frames(video_path, row[1:], output_folder) 
        extract_frames_at_times_FFmpeg(video_path, row[1:], output_folder)   # much faster      
        

# Specify the path to your Excel sheet
excel_file_path = "F:/Test/CV/src/preprocess/raw/flipped/label.xlsx"  # Replace with your Excel file path

output_folder = 'F:/Test/CV/src/preprocess/images/'

os.makedirs(output_folder, exist_ok=True)

skip_line_count = 5  # skip the first several lines in the excel file

# Call the function to process the videos
process_videos_from_excel(excel_file_path, output_folder, skip_line_count)