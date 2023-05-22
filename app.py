# from ImageGoNord import NordPaletteFile, GoNord
import ffmpeg
import numpy as np
import sys
from PIL import Image, ImageColor
import os
from datetime import datetime

def assemble_video(input_dir, output_dir = '.'):
    '''
    Assemble video from sequence of frames

    input_dir: Directory path of images. Images can be in other supported formats as well.

    output_dir: Saves video to location. Defaults to base directory.
    '''
    (
        ffmpeg
        .input(f'{input_dir}/*.jpg', pattern_type='glob', framerate=25)
        .output(f'{output_dir}/movie.gif', loglevel='quiet')
        .run()
    )

def convert_palette(palette, img_array):
    
    # if the image doesn't have an alpha channel, add one with all 255s
    if img_array.shape[2] == 3:
        img_array = np.concatenate((img_array, np.full((img_array.shape[0], img_array.shape[1], 1), 255)), axis=2)
    
    # create a new array to hold the pixelated image
    pixelated_array = np.zeros(img_array.shape)

    norms = np.linalg.norm(
        palette[np.newaxis, np.newaxis, :, :] 
        - img_array[:, :, np.newaxis, :], axis=-1
    )
    closest_indexes = np.argmin(norms, axis=-1)

    # loop over the image array
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            pixelated_array[i][j] = palette[closest_indexes[i][j]]
    
    return pixelated_array

def convert_vid_to_np_arr(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    out, _ = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel='quiet')
        .run(capture_stdout=True)
    )

    video_np_arr = (
        # np.array(img.resize((int(width/4), int(height/4)), Image.LANCZOS))
        # np
        # .frombuffer(out, np.uint8)
        # .reshape([-1, height, width, 3])
         np.array(
        np.frombuffer(out, np.uint8)
                 .reshape([-1, height, width, 3]))
    )

    return video_np_arr


def main():
    nord_palette = [
    "#BF616AFF",
    "#D08770FF",
    "#EBCB8BFF",
    "#A3BE8CFF",
    "#B48EADFF",
    "#8FBCBBFF",
    "#88C0D0FF",
    "#81A1C1FF",
    "#5E81ACFF",
    "#2E3440FF",
    "#3B4252FF",
    "#434C5EFF",
    "#4C566AFF",
    "#D8DEE9FF",
    "#E5E9F0FF",
    "#ECEFF4FF"
    ]

    # convert color palette to np array
    nord_palette = np.array(
        [np.array(ImageColor.getrgb(color)) for color in nord_palette]
    )
    start_time = datetime.now()

    np_arr = convert_vid_to_np_arr('video/luffy.gif')

    converted_array = np.zeros(
            (np_arr.shape[0],np_arr.shape[1],np_arr.shape[2],np_arr.shape[3] + 1)
        ) if np_arr[0].shape[2] == 3 else np.zeros(np_arr.shape)
    for ind, frame in enumerate(np_arr):
        converted_array[ind] = convert_palette(nord_palette, frame)
        (
            Image
            .fromarray((converted_array[ind] * 255).astype(np.uint8))
            .save(f'images/frame{str(ind).zfill(len(str(len(converted_array))))}.png')
        )
    # for ind in range(len(np_arr)):
    # nordify()
    # assemble_video('images')

    print('Duration: {}'.format(datetime.now() - start_time))

if __name__ == "__main__":
    main()