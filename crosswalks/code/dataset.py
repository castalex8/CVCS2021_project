import cv2
import sys
import argparse


def extractImages(pathIn, pathOut, numVideo):
    count = 0
    vidcap = cv2.VideoCapture(pathIn + "\\DREYEVE_DATA\\{}\\video_garmin.avi".format(numVideo))
    success, frame = vidcap.read()
    if not success:
        sys.exit("Error in accessing the video")
    while True:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 500))
        success, frame = vidcap.read()
        if not success:
            sys.exit("Error in accessing the video (end of video?)")
        print('Read a new frame:', success, "Frame:", count)
        ret = cv2.imwrite(filename=pathOut + "\\{}_{:03d}.png".format(numVideo, count), img=frame)
        if ret is not True:
            sys.exit("Error saving image")
        count += 1


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--numVideo", help="number of video for extraction: 01, 02, ..., 74")
    a.add_argument("--pathIn", help="first part of path")
    a.add_argument("--pathOut", help="absolute path to images")
    args = a.parse_args()
    if int(args.numVideo) <= 0 or int(args.numVideo) >= 75:
        sys.exit("Wrong number")
    print(args)
    extractImages(args.pathIn, args.pathOut, args.numVideo)
