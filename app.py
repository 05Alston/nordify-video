import ffmpeg
import numpy as np
from PIL import Image, ImageColor
from datetime import datetime

def paletteuse():
    (
        ffmpeg
        .filter(
            [
                ffmpeg.input('video/one-piece.mp4'), 
                ffmpeg.input('nord-palette.png')
            ],
            filter_name='paletteuse', 
            dither='none'
        )
        .output('paletteuse_none.mp4', loglevel='quiet')# , vframes=1, format='image2', vcodec='mjpeg'
        .run(overwrite_output=True)
    )

def assemble_video(input_dir, num_frames, output_dir = '.'):
    '''
    Assemble video from sequence of frames

    input_dir: Directory path of images. Images can be in other supported formats as well.

    output_dir: Saves video to location. Defaults to base directory.
    '''
    num_frames = len(str(num_frames))
    (
        ffmpeg
        .input(f'{input_dir}/frame%0{num_frames}d.jpg')
        .output(f'{output_dir}/movie.mp4', loglevel='quiet')
        .run()
    )

def convert_palette(palette, img_array):
    
    # if the image doesn't have an alpha channel, add one with all 255s
    # if img_array.shape[2] == 3:
    #     img_array = np.concatenate((img_array, np.full((img_array.shape[0], img_array.shape[1], 1), 255)), axis=2)
    
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

def clear_lines(lines = 1):
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for idx in range(lines):
        print(LINE_UP, end=LINE_CLEAR)

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
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, height, width, 3])
    )

    return video_np_arr


def main():
    paletteuse()
    return
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

    start_time = datetime.now()

    # convert color palette to np array
    nord_palette = np.array(
        [np.array(ImageColor.getrgb(color)) for color in nord_palette]
    )

    np_arr = convert_vid_to_np_arr('video/one-piece.mp4')

    for ind, frame in enumerate(np_arr):
        (
            Image
            .fromarray(
                (
                    convert_palette(
                        nord_palette, 
                        frame
                    )
                )
                .astype(np.uint8)
            )
            .convert('RGB')
            .save(f'images/frame{str(ind).zfill(len(str(len(np_arr))))}.jpg')
        )
        print(f'Processed: {ind + 1} / {len(np_arr)}')
        clear_lines()
    assemble_video('images', len(np_arr))

    print('Duration: {}'.format(datetime.now() - start_time))

if __name__ == "__main__":
    main()
