# nordify-video

A video palette converter inspired by Schrodinger-Hat's [ImageGoNord](https://github.com/Schrodinger-Hat/ImageGoNord)  

PS: Even in non-ffmpeg version, only the palette conversion algorithm is free of ffmpeg code. ffmpeg is currently being used to get the video stream and also to convert individual frames to video format. I am looking for a way to remove this dependency.

## Installation

### Requirements

- ffmpeg already installed
- Run the pip command `pip install ffmpeg-python numpy`
- Run `python app.py`

#### ffmpeg version

- Run `python app.py`

#### non-ffmpeg version

- Run `python noff.py`

## TODO

- Find a way to get video stream as numpy array in noff.py

## Example

|Original Video|  
|--|  
|![original video](assets/original.gif)|  

|ffmpeg converted version|  
|--|  
|![ffmpeg converted video](assets/ffmpeg.gif)|  

|Non ffmpeg converted version|  
|--|  
|![non ffmpeg converted video](assets/no_ffmpeg.gif)|  
