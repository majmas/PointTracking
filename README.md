# PointTracking

This code is written by Majid Masoumi @ dr.majid.masoumi@gmail.com

I have used lucas kanade optical flow technique to track the points between frames.

prerequisites:

Install the following packages before running the code

pip install numpy
pip install pandas
pip install glob2
pip install opencv-python
pip install fsspec

Inputs:

The function takes two inputs:

1- Path to the frames folder
2- Path to the csv file (frame_points_output.csv)


Output: 

The function output two csv files meanwhile allows to visually watch the tracking point among frames

1- The location of each point on every frame (points_location.csv)
2- The error of each point for every frame (points_error.csv)


To run the code:

1- Change the defult paths in line 108 and 109 to your local paths to frame folders and csv file.
2- Open Anaconda prompt
3- change the default path to the folder the contatins the trackingpoint.py (e.g. cd C:\Users\majid\PycharmProjects\trackpoints )
4- Type -->  python trackpoints.py
A motion tracking system for any arbitaray points in a video frame.
