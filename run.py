"""
file used to load/test pockit module
"""
import os
from pockit import pockit
from pockit.constants import data

video = os.path.join(data, 'file1_trimmed.mp4')
video = os.path.join(data, 'IMG_5662.MOV')
video = os.path.join(data, 'IMG_5662_trimmed_2.mp4')
video = os.path.join(data, 'output.mp4')
# video = os.path.join(data, 'face.mp4')

def main():
    print(pockit.get_info(video))
    pockit.load(video)


if __name__ == '__main__':
    main()

