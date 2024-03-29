# from moviepy.editor import VideoFileClip, concatenate_videoclips
from datetime import datetime
import numpy as np
import subprocess
import argparse
import uuid
import cv2
import os

def get_video_information(video_path):
    '''
    Get basic information about the video file.

    #@param video_path: relative/ absolute path of input video file
    '''
    ''' #? All possible properties of the video
    0. CAP_PROP_POS_MSEC : Current position of the video file in milliseconds.
    1. CAP_PROP_POS_FRAMES : 0-based index of the frame to be decoded/captured next.
    2. CAP_PROP_POS_AVI_RATIO : Relative position of the video file
    3. CAP_PROP_FRAME_WIDTH : Width of the frames in the video stream.
    4. CAP_PROP_FRAME_HEIGHT : Height of the frames in the video stream.
    5. CAP_PROP_FPS : Frame rate.
    6. CAP_PROP_FOURCC : 4-character code of codec.
    7. CAP_PROP_FRAME_COUNT : Number of frames in the video file.
    8. CAP_PROP_FORMAT : Format of the Mat objects returned by retrieve() .
    9. CAP_PROP_MODE : Backend-specific value indicating the current capture mode.
    10. CAP_PROP_BRIGHTNESS : Brightness of the image (only for cameras).
    11. CAP_PROP_CONTRAST : Contrast of the image (only for cameras).
    12. CAP_PROP_SATURATION : Saturation of the image (only for cameras).
    13. CAP_PROP_HUE : Hue of the image (only for cameras).
    14. CAP_PROP_GAIN : Gain of the image (only for cameras).
    15. CAP_PROP_EXPOSURE : Exposure (only for cameras).
    16. CAP_PROP_CONVERT_RGB : Boolean flags indicating whether images should be converted to RGB.
    17. CAP_PROP_WHITE_BALANCE : Currently unsupported
    18. CAP_PROP_RECTIFICATION : Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
    '''
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps

    cap.release()

    return width, height, fps, duration, total_frames

def convert_vid_to_np_arr(video_path, cube, start_frame, duration):
    '''
    Convert video to array of numpy elements.

    #@param video_path: relative/ absolute path of input video file

    #@param cube: color cube created from the palette

    #@param start_time: time to seek forward in the video

    #@param duration: number of frames to capture
    '''

    frames = []
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    curr = 0
    success, image = cap.read()

    while success and curr < duration:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = convert_palette(cube, image)
        frames.append(image)
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"Frame: {frame_num}/{total_frames}")
        clear_lines()
        curr += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + curr) 
        success, image = cap.read()
    cap.release()
    return (frames)

def convert_palette(color_cube, image):
    '''
    Convert each frame to desired color palette.

    #@param color_cube: color cube created from the palette

    #@param image: current frame.
    '''
    shape = image.shape[0:2]
    indices = image.reshape(-1,3)
    # Pass image colors and retrieve corresponding palette color
    new_image = color_cube[indices[:,0],indices[:,1],indices[:,2]]

    return new_image.reshape(shape[0],shape[1],3).astype(np.uint8)

def generate_color_map(palette, palette_name):
    '''
    Generate a color cube.

    #@param palette: numpy array which contains the complete color palette.

    #@param palette_name: name of the color palette.
    '''
    precalculated = np.zeros(shape=[256,256,256,3])
    for i in range(256):
        print(f"building color palette: %0.2f%%" %(100 * i / 256))
        clear_lines()
        for j in range(256):
            for k in range(256):
                index = np.argmin(np.sqrt(np.sum(
                        ((palette)-np.array([i,j,k]))**2,
                        axis=1
                    )))
                precalculated[i,j,k] = palette[index]
    print("building color palette: 100%")
    np.savez_compressed(palette_name, color_cube = precalculated)

def write_video(filename, vid_list, width, height, fps):
    '''
    Generate video from the numpy array.

    #@param filename: filename
    
    #@param vid_list: list of video frames
    
    #@param width: width of video frame

    #@param height: height of video frame

    #@param fps: framerate of the video.
    '''
    codec_id = "mp4v" # ID for a video codec.
    fourcc = cv2.VideoWriter_fourcc(*codec_id)
    out = cv2.VideoWriter(filename, fourcc=fourcc, fps=fps, frameSize=(width, height))

    for frame in vid_list:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()

def concat_video(uid, out):
    '''
    Concatenate two videos.

    #@param uid: Unique identifier for the session

    #@param out: Output video
    '''
    command = [
        "ffmpeg",
        "-f", "concat",
        "-i", f"vids_{uid}.txt",
        "-c", "copy",
        "-loglevel", "quiet",
        "-y",
        f"output_{uid}.mp4"
    ]
    subprocess.run(command)
    #* Moviepy implementation of concatenating videos
    # clip1 = VideoFileClip(out)
    # clip2 = VideoFileClip(f"temp_{uid}.mp4")
    # final_clip = concatenate_videoclips([clip1,clip2])
    # final_clip.write_videofile(f"output_{uid}.mp4", logger=None)
    os.remove(f"temp_{uid}.mp4")
    os.remove(out)
    os.rename(f"output_{uid}.mp4", out)

def clear_lines(lines = 1):
    '''
    Clear the last 'n' lines

    #@param lines: Number of terminal lines to go up.
    '''
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    for _ in range(lines):
        print(LINE_UP, end=LINE_CLEAR)

def main(_input, _output):
    # Generate some random unique identifier that is generated for each session for the temporary files.
    uid = uuid.uuid4()

    start_time = datetime.now()
    nord_palette = np.array(
        [[46, 52,  64], # nord 0
        [59, 66,  82], # nord 1
        [67, 76,  94], # nord 2
        [76, 86, 106], # nord 3
        [216, 222, 233], # nord 4
        [229, 233, 240], # nord 5
        [236, 239, 244], # nord 6
        [143, 188, 187], # nord 7
        [136, 192, 208], # nord 8
        [129, 161, 193], # nord 9
        [94, 129, 172], # nord 10
        [191, 97, 106], # nord 11
        [208, 135, 112], # nord 12
        [235, 203, 139], # nord 13
        [163, 190, 140], # nord 14
        [180, 142, 173]] # nord 15
    )

    palette_name = "nord"

    width, height, framerate, duration, total_frames = get_video_information(_input)

    # Create a file that contains the names of the two video files for concatenation
    f = open(f"vids_{uid}.txt", "w")
    f.write(f"file '{_output}'\n")
    f.write(f"file 'temp_{uid}.mp4'\n")
    f.close()

    print("####VIDEO INFORMATION#####")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"FPS: {framerate}")
    print(f"Duration: {duration} s")
    try:
        # for all colors (256*256*256) assign color from palette
        precalculated = np.load(f"{palette_name}.npz")["color_cube"]
    except:
        generate_color_map(nord_palette, palette_name)
    
    frame_number = 0
    frames_per_batch = 400
    while frame_number < total_frames:
        frame_list = convert_vid_to_np_arr(_input, precalculated, frame_number, frames_per_batch)
        if os.path.exists(_output):
            write_video(f"temp_{uid}.mp4", frame_list, width, height, framerate)
            concat_video(uid, _output)
        else:
            write_video(_output, frame_list, width, height, framerate)
        if (total_frames - frame_number) < frames_per_batch:
            frames_per_batch = total_frames - frame_number
        frame_number += frames_per_batch
    os.remove(f"vids_{uid}.txt")
    print(f"Total running duration: {datetime.now() - start_time}")
    return

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("input", metavar="input", type=str, help="input filename")
    a.add_argument("-o", "--output", metavar="output", type=str, help="output filename", default="movie.mp4")
    args = a.parse_args()
    _input = args.input
    _output = args.output
    main(_input, _output)
