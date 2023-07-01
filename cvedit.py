import argparse
import numpy as np
import cv2
import os
import resource
from moviepy.editor import VideoFileClip, concatenate_videoclips

def get_video_information(video_path):
    '''
    Get basic information about the video file.

    #@param video_path: Relative/Absolute path of input video file
    '''
    ''' #! All possible properties of the video
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

def convert_vid_to_np_arr(video_path, cube, width, height, start_frame, duration):
    '''
    Convert video to array of numpy elements.

    video_path: Relative/Absolute path of input video file

    width: Width of video(numpy array width)

    height: Height of video(numpy array depth)

    start_time: Time to seek forward in the video

    duration: Number of frames to capture
    '''

    frames = []

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    curr = 0
    success, image = cap.read()

    while success and curr < duration:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = convert_palette(cube, image)
        frames.append(image)
        print(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        clear_lines()
        curr += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + curr) 
        success, image = cap.read()
    cap.release()
    return (frames)

def convert_palette(color_cube, image):
    '''
    Convert each frame to desired color palette.

    color_cube: Color cube created from the palette

    image: Current frame.
    '''
    shape = image.shape[0:2]
    indices = image.reshape(-1,3)
    # Pass image colors and retrieve corresponding palette color
    new_image = color_cube[indices[:,0],indices[:,1],indices[:,2]]

    return new_image.reshape(shape[0],shape[1],3).astype(np.uint8)

def generate_color_map(palette, palette_name):
    '''
    Generate a color cube.

    palette: Numpy array which contains the complete color palette.

    palette_name: Name of the color palette.
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
    print('building color palette: 100%')
    np.savez_compressed(palette_name, color_cube = precalculated)

def write_video(filename, vid_list, width, height, fps):
    
    codec_id = "mp4v" # ID for a video codec.
    fourcc = cv2.VideoWriter_fourcc(*codec_id)
    out = cv2.VideoWriter(filename, fourcc=fourcc, fps=20, frameSize=(width, height))

    for frame in vid_list:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()

def concat_video(vid1, vid2, _output):
    clip1 = VideoFileClip(vid1)
    clip2 = VideoFileClip(vid2)
    final_clip = concatenate_videoclips([clip1,clip2])
    final_clip.write_videofile(_output)

#def concat_video(vid1, vid2, _output):
#    command = [
#            'ffmpeg',
#            '-f', 'concat',
#            '-i', 'list.txt',
#            '-c', 'copy',
#            'out.mp4'
#            ]
#    subprocess.run(command)

def clear_lines(lines = 1):
    '''
    Clear the last 'n' lines

    lines: Number of terminal lines to go up.
    '''
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(lines):
        print(LINE_UP, end=LINE_CLEAR)

def main(_input):
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

    palette_name = 'nord'

    width, height, framerate, duration, total_frames = get_video_information(_input)
    #print(width, height, framerate, duration, total_frames)
    print('####VIDEO INFORMATION#####')
    print(f'Width: {width}')
    print(f'Height: {height}')
    print(f'FPS: {framerate}')
    print(f'Duration: {duration} s')
    try:
        # for all colors (256*256*256) assign color from palette
        precalculated = np.load(f'{palette_name}.npz')['color_cube']
    except:
        generate_color_map(nord_palette, palette_name)
    frame_number = 0
    frames_per_batch = 100
    while frame_number < total_frames:
        frame_list = convert_vid_to_np_arr(_input, precalculated, width, height, frame_number, frames_per_batch)
        #print(np_array.size * np_array.itemsize)
        if os.path.exists('out.mp4'):
            write_video('temp.mp4', frame_list, width, height, framerate)
            concat_video('out.mp4', 'temp.mp4', 'output.mp4')
            return
        else:
            write_video('out.mp4', frame_list, width, height, framerate)
            #return
        if (total_frames - frame_number) < frames_per_batch:
            frames_per_batch = total_frames - frame_number
        frame_number += frames_per_batch

    return

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("input", metavar='input', type=str, help="input filename")
    args = a.parse_args()
    _input = args.input
    main(_input)
