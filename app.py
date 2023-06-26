import ffmpeg
from datetime import datetime
import argparse

def paletteuse(_input, _output):
    (
        ffmpeg
        .filter(
            [
                ffmpeg.input(_input), 
                ffmpeg.input('nord-palette.png')
            ],
            filter_name='paletteuse', 
            dither='none'
        )
        .output(_output, loglevel='quiet')
        .run(overwrite_output=True)
    )

if __name__ == "__main__":
    start_time = datetime.now()
    a = argparse.ArgumentParser()
    a.add_argument("input", metavar="input", type=str, help="input filename")
    a.add_argument("-o", "--output", metavar="output", type=str, help="output filename", default="movie_no_ffmpeg.mp4")
    args = a.parse_args()
    input = args.input
    output = args.output
    paletteuse(input, output)
    print('Duration: {}'.format(datetime.now() - start_time))