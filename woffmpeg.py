import ffmpeg
import numpy as np
from datetime import datetime
import argparse
from PIL import Image
import subprocess as sp
import resource
import sys

def get_video_information(video_path):
    '''
    Get basic information about the video file.

    video_path: Relative/Absolute path of input video file
    '''
    probe = ffmpeg.probe(video_path)
    video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
            None)

    width = int(video_stream['width'])
    height = int(video_stream['height'])
    framerate = video_stream['avg_frame_rate'].split('/')[0]
    duration = video_stream['duration']
    total_frames = video_stream['nb_frames']

    return int(width), int(height), int(framerate), float(duration), int(total_frames)

def convert_vid_to_np_arr(video_path, width, height, start_time, duration):
    '''
    Convert video to array of numpy elements.

    video_path: Relative/Absolute path of input video file

    width: Width of video(numpy array width)

    height: Height of video(numpy array depth)

    start_time: Time to seek forward in the video

    duration: Number of frames to capture
    '''
    command = [ 'ffmpeg',
            '-ss', str(start_time), # seek time
            '-i', video_path, # input path
            '-pix_fmt', 'rgb24', # pixel format
            '-f', 'rawvideo', # video format
            '-t', str(duration), # duration/ number of frames
            '-loglevel', 'quiet', # log level
            'pipe:' ] # output

    # Run the above command and output the result to stdout
    process = sp.run(command, stdout=sp.PIPE, bufsize=10**8)

    # Generate numpy array from stdout
    video_np_arr = (
        np
        .frombuffer(process.stdout, dtype = np.uint8)
        .reshape([-1, height, width, 3])
    )

    return video_np_arr

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

def assemble_video(input_dir, num_frames, framerate, output):
    num_frames = len(str(num_frames))

    command = ['ffmpeg',
               '-framerate', str(framerate),
               '-i', f'{input_dir}/frame%0{num_frames}d.jpg',
               '-pix_fmt', 'yuv420p',
               '-c:v', 'libx264',
               '-y',
               '-loglevel', 'quiet',
               output]

    sp.run(command)

def clear_lines(lines = 1):
    '''
    Clear the last 'n' lines
    '''
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for idx in range(lines):
        print(LINE_UP, end=LINE_CLEAR)

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

def main(_input, _output):
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

    palette_name = 'nord'
    # run once to generate the color map file
    try:
        # for all colors (256*256*256) assign color from palette
        precalculated = np.load(f'{palette_name}.npz')['color_cube']
    except:
       generate_color_map(nord_palette, palette_name)

    # Initialize variables for conversion
    width, height, framerate, duration, total_frames = get_video_information(_input)
    frames_per_batch = 300
    ind = 0
    timestamp = 0
    batch_dur = frames_per_batch / framerate
    batch_dur = batch_dur if duration > batch_dur else duration
    print('####VIDEO INFORMATION#####')
    print(f'Width: {width}')
    print(f'Height: {height}')
    print(f'Duration: {duration} s')
    print(f'Processed: {ind} / {total_frames} frames')

    # Process the entire video in batches of `frames_per_batch` frames
    while ind < total_frames:
        np_arr = convert_vid_to_np_arr(_input, width, height, timestamp, batch_dur)

        for frame in np_arr:
            (
                Image.fromarray(
                    convert_palette(precalculated, frame)
                    )
                .convert('RGB')
                .save(f'images/frame{str(ind).zfill(len(str(total_frames)))}.jpg')
                )
            ind += 1
            clear_lines()
            print(f'Processed: {ind} / {total_frames} frames')
        duration -= batch_dur
        timestamp += batch_dur 
        batch_dur = batch_dur if duration > batch_dur else duration

    assemble_video('images', total_frames, framerate, _output)
    print(f'Memory used: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024} Mbs') # Mbs
    print(f'Total running duration: {datetime.now() - start_time}')

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("input", metavar='input', type=str, help="input filename")
    a.add_argument("-o", "--output", metavar='output', type=str, help="output filename", default='movie.mp4')
    args = a.parse_args()
    _input = args.input
    _output = args.output
    main(_input, _output)

