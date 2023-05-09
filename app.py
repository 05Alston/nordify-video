from ImageGoNord import NordPaletteFile, GoNord
import ffmpeg

#list_imgs = os.listdir('images')


def assemble_video(input_dir, output_dir = '.'):
    '''
    Assemble video from sequence of frames

    input_dir: Directory path of images. Images can be in other supported formats as well.

    output_dir: Saves video to location. Defaults to base directory.
    '''
    ffmpeg
    .input(f'{input_dir}/*.jpg', pattern_type='glob', framerate=25)
    .output(f'{output_dir}/movie.mp4')
    .run()


def nordify(img_dir = 'images'):
    '''
    img_dir: location of all frames.

    Replace image with image converted to nord palette.
    '''
    print("Nordifying Images")
    go_nord = GoNord()
    img_list = os.listdir(img_dir)
    for img in img_list:
        image = go_nord.open_image(os.path.join(img_dir, img))
        go_nord.convert_image(image, save_path=os.path.join(img_dir, img))
    print("Successfully Nordified")

'''

# Flip horizontally
(
    ffmpeg
    .input('video/tom.mp4')
    .hflip()
    .output('output.mp4')
    .run()
)
'''
'''
probe = ffmpeg.probe('video/tom.mp4')
video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
width = int(video_stream['width'])
height = int(video_stream['height'])
'''















#### Old stuff
'''
print("Frame Extraction Begun")
vidcap = cv2.VideoCapture('video/tom.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(f'images/image{str(count)}.jpg', image) # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 1 / 12 # 12 FPS
count=1
success = getFrame(sec)

while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

print("Frame Extraction Finished")
print("Nordifying Images")
go_nord = GoNord()
for img in list_imgs:
    image = go_nord.open_image(os.path.join('images', img))
    go_nord.convert_image(image, save_path=f'images/{img}')

print("Successfully Nordified")
print("Converting to video")

img = []
for image in list_imgs:
    img.append(cv2.imread(os.path.join('images', image)))

height,width,layers=img[1].shape

video=cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'MP42'),1/12,(width,height))
for image in img:
    video.write(image)

cv2.destroyAllWindows()
video.release()
'''