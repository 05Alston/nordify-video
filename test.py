from moviepy.editor import VideoFileClip, concatenate_videoclips

clip1 = VideoFileClip('out.mp4')
clip2 = VideoFileClip('temp.mp4')
final_clip = concatenate_videoclips([clip1,clip2])
final_clip.write_videofile('output.mp4', logger=None)