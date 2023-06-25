import ffmpeg
import numpy as np
from PIL import Image
from datetime import datetime

def convert_vid_to_np_arr(video_path):
    '''
    Convert video to array of numpy elements  
    '''
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

def convert_palette(color_cube,image):
    '''
    Convert each frame to desired color palette.

    color_cube: 

    image: Current frame.
    '''
    shape = image.shape[0:2]
    indices = image.reshape(-1,3)
    # pass image colors and retrieve corresponding palette color
    new_image = color_cube[indices[:,0],indices[:,1],indices[:,2]]
   
    return new_image.reshape(shape[0],shape[1],3).astype(np.uint8)

def assemble_video(input_dir, num_frames, output_dir = '.'):
    '''
    Assemble video from sequence of frames
    '''
    num_frames = len(str(num_frames))
    (
        ffmpeg
        .input(f'{input_dir}/frame%0{num_frames}d.jpg')
        .output(f'{output_dir}/movie.mp4', loglevel='quiet')
        .run()
    )

def clear_lines(lines = 1):
    ''' Clear the last 'n' lines '''
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for idx in range(lines):
        print(LINE_UP, end=LINE_CLEAR)

def main():
    nord_palette = np.array([[191, 97, 106],
        [208, 135, 112],
        [235, 203, 139],
        [163, 190, 140],
        [180, 142, 173],
        [143, 188, 187],
        [136, 192, 208],
        [129, 161, 193],
        [94, 129, 172],
        [46, 52,  64],
        [59, 66,  82],
        [67, 76,  94],
        [76, 86, 106],
        [216, 222, 233],
        [229, 233, 240],
        [236, 239, 244]])

    start_time = datetime.now()
    try:
        # for all colors (256*256*256) assign color from palette
        precalculated = np.load('view.npz')['color_cube']
    except:
        precalculated = np.zeros(shape=[256,256,256,3])
        for i in range(256):
            print("processing %0.2f" %(100 * i / 256))
            clear_lines()
            for j in range(256):
                for k in range(256):
                    index = np.argmin(np.sqrt(np.sum(((nord_palette)-np.array([i,j,k]))**2,axis=1)))
                    precalculated[i,j,k] = nord_palette[index]
        print('processing 100%')
        np.savez_compressed('view', color_cube = precalculated)

    np_arr = convert_vid_to_np_arr('video/one-piece.mp4')

    for ind, frame in enumerate(np_arr):
        (
            Image
            .fromarray(
                    convert_palette(
                        precalculated, 
                        frame
                    )
            )
            .convert('RGB')
            .save(f'images/frame{str(ind).zfill(len(str(len(np_arr))))}.jpg')
        )
        print(f'Processed: {ind + 1} / {len(np_arr)} frames')
        clear_lines()
    print(f'Processed: {len(np_arr)} / {len(np_arr)} frames')
    assemble_video('images', len(np_arr))
    print('Duration: {}'.format(datetime.now() - start_time))

if __name__ == "__main__":
    main()
