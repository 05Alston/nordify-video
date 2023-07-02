from datetime import datetime
import numpy as np
import subprocess
import argparse
import ffmpeg
import uuid
import os

def get_video_information(video_path):
    '''
    Get basic information about the video file.

    #@param video_path: Relative/Absolute path of input video file
    '''
    probe = ffmpeg.probe(video_path)
    video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None)

    width = int(video_stream["width"])
    height = int(video_stream["height"])
    framerate = int(video_stream["avg_frame_rate"].split("/")[0])
    duration = float(video_stream["duration"])
    total_frames = int(video_stream["nb_frames"])

    return width, height, framerate, duration, total_frames

def convert_vid_to_np_arr(video_path, width, height, start_time, duration):
    '''
    Convert video to array of numpy elements.

    #@param video_path: Relative/Absolute path of input video file

    #@param width: Width of video(numpy array width)

    #@param height: Height of video(numpy array depth)

    #@param start_time: Time to seek forward in the video

    #@param duration: Number of frames to capture
    '''
    command = [ "ffmpeg",
            "-ss", str(start_time), # seek time
            "-i", video_path, # input path
            "-pix_fmt", "rgb24", # pixel format
            "-f", "rawvideo", # video format
            "-t", str(duration), # duration/ number of frames
            "-loglevel", "quiet", # log level
            "pipe:" ] # output

    # Run the above command and output the result to stdout
    process = subprocess.run(command, stdout=subprocess.PIPE, bufsize=10**8)

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

    #@param color_cube: Color cube created from the palette

    #@param image: Current frame.
    '''
    shape = image.shape[0:2]
    indices = image.reshape(-1,3)
    # Pass image colors and retrieve corresponding palette color
    new_image = color_cube[indices[:,0],indices[:,1],indices[:,2]]

    return new_image.reshape(shape[0],shape[1],3).astype(np.uint8)

def generate_color_map(palette, palette_name):
    '''
    Generate a color cube.

    #@param palette: Numpy array which contains the complete color palette.

    #@param palette_name: Name of the color palette.
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

def vidwrite(fn, cube, images, framerate, start_frame, total_frames, vcodec="libx264"):
    '''
    Generate video from the numpy array.

    #@param fn: Filename.
    
    #@param cube: color map that is generated.
    
    #@param images: (Numpy array / normal list) of frames.
    
    #@param framerate: FPS of the video.

    #@param v_codec: Video codec of the output.
    '''
    # If images is a list, convert to ndarray
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    height, width = images.shape[1:3]
    process = (
        ffmpeg
            .input("pipe:", format="rawvideo", pix_fmt="rgb24", s="{}x{}".format(width, height))
            .output(fn, pix_fmt="yuv420p", vcodec=vcodec, r=framerate, loglevel="quiet")
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for idx, frame in enumerate(images):
        clear_lines()
        print(f"Frame: {start_frame + idx + 1}/{total_frames}")
        process.stdin.write(
            convert_palette(cube, frame)
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()

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
    # run once to generate the color map file
    try:
        # for all colors (256*256*256) assign color from palette
        precalculated = np.load(f"{palette_name}.npz")["color_cube"]
    except:
       generate_color_map(nord_palette, palette_name)

    # Initialize variables for conversion
    width, height, framerate, duration, total_frames = get_video_information(_input)

    frames_per_batch = 100
    frame_number = 0
    timestamp = 0
    batch_dur = frames_per_batch / framerate
    batch_dur = batch_dur if duration > batch_dur else duration
    print("####VIDEO INFORMATION#####")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"FPS: {framerate}")
    print(f"Duration: {duration} s\n")
    print(f"Processed: {frame_number} / {total_frames} frames")

    # Create a file that contains the names of the two video files for concatenation
    f = open(f"vids_{uid}.txt", "w")
    f.write(f"file '{_output}'\n")
    f.write(f"file 'temp_{uid}.mp4'\n")
    f.close()

    # Process the entire video in batches of `frames_per_batch` frames
    while frame_number < total_frames:
        np_arr = convert_vid_to_np_arr(_input, width, height, timestamp, batch_dur)
        if os.path.exists(_output):
            vidwrite(f"temp_{uid}.mp4", precalculated, np_arr, framerate, frame_number, total_frames)
            concat_video(uid, _output)
        else:
            vidwrite(_output, precalculated, np_arr, framerate, frame_number, total_frames)
        if (total_frames - frame_number) < frames_per_batch:
            frames_per_batch = total_frames - frame_number
        frame_number += frames_per_batch
        duration -= batch_dur
        timestamp += batch_dur 
        batch_dur = batch_dur if duration > batch_dur else duration
    # print(f"Memory used: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024} Mbs")

    # Remove the temporary files
    if os.path.exists(f"temp_{uid}.mp4"):
        os.remove(f"temp_{uid}.mp4")
    os.remove(f"vids_{uid}.txt")
    print(f"Total running duration: {datetime.now() - start_time}")

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("input", metavar="input", type=str, help="input filename")
    a.add_argument("-o", "--output", metavar="output", type=str, help="output filename", default="movie.mp4")
    args = a.parse_args()
    _input = args.input
    _output = args.output
    main(_input, _output)