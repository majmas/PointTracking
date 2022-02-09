import numpy as np
import cv2
import glob
import csv
import pandas as pd


def trackpoints(input_frames, input_csv_file):

    # read input video frames
    framecap = []
    for filename in glob.glob(input_frames):
        img = cv2.imread(filename)
        framecap.append(img)

    # decide what column of CSV file needed to be analyzed
    cols = pd.read_csv(input_csv_file, nrows=1).columns
    df = pd.read_csv(input_csv_file, usecols=cols[1:])

    # adjust the size of input points to feed to lucas kanade
    p0 = df.values.reshape(-1, 1, 2).astype('float32')

    # Parameters for lucas kanade optical flow
    tracking_params = dict(winSize=(80, 80),
                           maxLevel=1,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors for visualization of tracking points
    color = np.random.randint(0, 255, (len(p0), 3))

    # converting list to np array to deal with requirement of lucas kanade
    cap = np.array(framecap, dtype='uint8')

    old_frame = cap[0, :, :, :]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # preallocate distance (error) matrix values
    dist = np.zeros([len(p0), 1], dtype=np.float32)

    # save the locations of point as well as distance error as csv files
    with open('points_location.csv', mode='w', newline='') as f, open('points_error.csv', mode='w', newline='') as g:
        keys = ['frame', 'p1_x', 'p1_y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y']
        writer = csv.DictWriter(f, fieldnames=keys, delimiter=',', lineterminator='\n')
        writer.writeheader()  # add column names in the CSV file

        keys1 = ['frame', 'p1', 'p2', 'p3', 'p4']
        writer1 = csv.DictWriter(g, fieldnames=keys1, delimiter=',', lineterminator='\n')
        writer1.writeheader()  # add column names in the CSV file

        for j in range(len(cap)):
            frame = cap[j, :, :, :]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **tracking_params)

            writer.writerow({'frame': "{:04d}".format(j + 1) + '.png', 'p1_x': p1[0, 0, 0],
                             'p1_y': p1[0, 0, 1], 'p2_x': p1[1, 0, 0], 'p2_y': p1[1, 0, 1], 'p3_x': p1[2, 0, 0],
                             'p3_y': p1[2, 0, 1], 'p4_x': p1[3, 0, 0], 'p4_y': p1[3, 0, 1]})
            # check error of optical flow
            # --------------------------
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(frame_gray, old_gray, p1, None, **tracking_params)

            # calculate the error of estimation
            for i in range(len(p1)):
                dist[i] = l1_distance(p0[i, ...], p0r[i, ...])

            writer1.writerow({'frame': "{:04d}".format(j + 1) + '.png',
                              'p1': dist[0], 'p2': dist[1], 'p3': dist[2], 'p4': dist[3]})

            # Select good points- points that are not out of frame
            # status - Indicates positive tracks. 1 = PosTrack 0 = NegTrack
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)

            cv2.imshow('frame', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        cv2.destroyAllWindows()


def l1_distance(a, b):
    d = np.zeros(1, dtype='float32')
    for i in range(len(a)):
        d += sum(abs(a[i] - b[i]))
    return d


if __name__ == '__main__':

    frame_path = 'C://Users//majid//PycharmProjects//trackpoints//frames//'
    csv_file_path = 'C://Users//majid//PycharmProjects//trackpoints//'

    input_frames = frame_path + '*.png'
    input_csv_file = csv_file_path + 'frame_points_output.csv'

    trackpoints(input_frames, input_csv_file)
