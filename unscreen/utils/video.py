import subprocess

import cv2


def get_numframes(video_path):
    """get the number of frames of a video.

    Args:
        video_path (str): the path of the video

    Returns:
        numframes (int): the number of frames
    """
    cap = cv2.VideoCapture(video_path)
    numframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return numframes


def get_video_size(video_path):
    """get the resolution of a video.

    Args:
        video_path (str): the path of the video

    Returns:
        height (int): the height of the video
        width (int): the width of the video
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return height, width


def get_video_duration(video_path):
    """get the duration of a video (in seconds)

    Args:
        video_path (str): the path of the video

    Returns:
        duration (float): the duration of the video (in seconds)
    """
    cmd = ('ffprobe -v error -select_streams v:0 '
           '-show_entries stream=duration '
           '-of default=noprint_wrappers=1:nokey=1')
    duration = subprocess.check_output(cmd.split() + [video_path])
    duration = float(duration.strip())
    return duration
