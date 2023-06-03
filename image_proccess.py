"""
Library for detecting a blob based on a color range filter in HSV space
   0------------------> x (cols)
   |
   |
   |         o center
   |
   |
   V y (rows)
"""
import time

# Standard imports
import numpy as np
# import serial

import cv2


COUNT = 100

# ---------- Blob detecting function: returns keypoints and mask
# -- return keypoints, reversemask
ser = serial.Serial('COM14', 9600)
#ser.open()
def blob_detect(image,  # -- The frame (cv standard)
                hsv_min,  # -- minimum threshold of the hsv filter [h_min, s_min, v_min]
                hsv_max,  # -- maximum threshold of the hsv filter [h_max, s_max, v_max]
                blur=0,  # -- blur value (default 0)
                blob_params=None,  # -- blob parameters (default None)
                search_window=None,
                # -- window where to search as [x_min, y_min, x_max, y_max] adimensional (0.0 to 1.0) starting from top left corner
                imshow=False
                ):
    # - Blur image to remove noise
    if blur > 0:
        image = cv2.blur(image, (blur, blur))
        # - Show result
        if imshow:
            cv2.imshow("Blur", image)
            cv2.waitKey(0)

    # - Search window
    if search_window is None: search_window = [0.0, 0.0, 1.0, 1.0]

    # - Convert image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # - Apply HSV threshold
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # - Show HSV Mask
    if imshow:
        cv2.imshow("HSV Mask", mask)

    # - dilate makes the in range areas larger
    mask = cv2.dilate(mask, None, iterations=2)
    # - Show HSV Mask
    if imshow:
        cv2.imshow("Dilate Mask", mask)
        cv2.waitKey(0)

    mask = cv2.erode(mask, None, iterations=2)

    # - Show dilate/erode mask
    if imshow:
        cv2.imshow("Erode Mask", mask)
        cv2.waitKey(0)

    # - Cut the image using the search mask
    mask = apply_search_window(mask, search_window)

    if imshow:
        cv2.imshow("Searching Mask", mask)
        cv2.waitKey(0)

    # - build default blob detection parameters, if none have been provided
    if blob_params is None:
        # Set up the SimpleBlobdetector with default parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0;
        params.maxThreshold = 100;

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 20000

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.4

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

    else:
        params = blob_params

        # - Apply blob detection
    detector = cv2.SimpleBlobDetector_create(params)

    # Reverse the mask: blobs are black on white
    reversemask = 255 - mask

    if imshow:
        cv2.imshow("Reverse Mask", reversemask)
        cv2.waitKey(0)

    keypoints = detector.detect(reversemask)

    return keypoints, reversemask


# ---------- Draw detected blobs: returns the image
# -- return(im_with_keypoints)
def draw_keypoints(image,  # -- Input image
                   keypoints,  # -- CV keypoints
                   line_color=(0, 0, 255),  # -- line's color (b,g,r)
                   imshow=False  # -- show the result
                   ):
    # -- Draw detected blobs as red circles.
    # -- cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), line_color,
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if imshow:
        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)

    return (im_with_keypoints)


# ---------- Draw search window: returns the image
# -- return(image)
def draw_window(image,  # - Input image
                window_adim,  # - window in adimensional units
                color=(255, 0, 0),  # - line's color
                line=5,  # - line's thickness
                imshow=False  # - show the image
                ):
    rows = image.shape[0]
    cols = image.shape[1]

    x_min_px = int(cols * window_adim[0]*-1)
    y_min_px = int(rows * window_adim[1]*-1)
    x_max_px = int(cols * window_adim[2]*1.5)
    y_max_px = int(rows * window_adim[3]*1.5)

    # -- Draw a rectangle from top left to bottom right corner
    image = cv2.rectangle(image, (x_min_px, y_min_px), (x_max_px, y_max_px), color, line)

    if imshow:
        # Show keypoints
        cv2.imshow("Keypoints", image)

    return (image)


# ---------- Draw X Y frame
# -- return(image)
def draw_frame(image,
               dimension=0.3,  # - dimension relative to frame size
               line=2  # - line's thickness
               ):
    rows = image.shape[0]
    cols = image.shape[1]
    size = min([rows, cols])
    center_x = int(cols / 2.0)
    center_y = int(rows / 2.0)

    line_length = int(size * dimension)

    # -- X
    image = cv2.line(image, (center_x, center_y), (center_x + line_length, center_y), (0, 0, 255), line)
    # -- Y
    image = cv2.line(image, (center_x, center_y), (center_x, center_y + line_length), (0, 255, 0), line)

    return (image)


# ---------- Apply search window: returns the image
# -- return(image)
def apply_search_window(image, window_adim=[0.0, 0.0, 1.0, 1.0]):
    rows = image.shape[0]
    cols = image.shape[1]
    x_min_px = int(cols * window_adim[0]*0.1)
    y_min_px = int(rows * window_adim[1]*0.1)
    x_max_px = int(cols * window_adim[2]*2)
    y_max_px = int(rows * window_adim[3]*2)

    # --- Initialize the mask as a black image
    mask = np.zeros(image.shape, np.uint8)

    # --- Copy the pixels from the original image corresponding to the window
    mask[y_min_px:y_max_px, x_min_px:x_max_px] = image[y_min_px:y_max_px, x_min_px:x_max_px]

    # --- return the mask
    return (mask)


# ---------- Apply a blur to the outside search region
# -- return(image)
def blur_outside(image, blur=5, window_adim=[0.0, 0.0, 1.0, 1.0]):
    rows = image.shape[0]
    cols = image.shape[1]
    x_min_px = int(cols * window_adim[0]*0)
    y_min_px = int(rows * window_adim[1]*0)
    x_max_px = int(cols * window_adim[2]*0)
    y_max_px = int(rows * window_adim[3]*0)

    # --- Initialize the mask as a black image
    mask = cv2.blur(image, (blur, blur))

    # --- Copy the pixels from the original image corresponding to the window
    mask[y_min_px:y_max_px, x_min_px:x_max_px] = image[y_min_px:y_max_px, x_min_px:x_max_px]

    # --- return the mask
    return (mask)


# ---------- Obtain the camera relative frame coordinate of one single keypoint
# -- return(x,y)
def get_blob_relative_position(image, keyPoint):
    rows = float(image.shape[0])
    cols = float(image.shape[1])
    # print(rows, cols)
    center_x = 0.5 * cols
    center_y = 0.5 * rows
    # print(center_x)
    x = (keyPoint.pt[0] - center_x) / (center_x)
    y = (keyPoint.pt[1] - center_y) / (center_y)
    return (x, y)


# ----------- TEST
if _name_ == "_main_":


    # --- Define HSV limits

   # blue_min = (0, 156, 206)
   # blue_max = (68,255, 255)
    blue_min = (73, 47, 0)
    blue_max = (97, 255,196)

    # --- Define area limit [x_min, y_min, x_max, y_max] adimensional (0.0 to 1.0) starting from top left corner
    window = [0.25, 0.25, 0.65, 0.75]

    # -- IMAGE_SOURCE: either 'camera' or 'imagelist'
    # SOURCE = 'video'
    SOURCE = 'video'

    if SOURCE == 'video':
        url = "http://192.168.1.11:4747/video"
        cap = cv2.VideoCapture(url)

        while (True):

            # Capture frame-by-frame
            ret, frame = cap.read()

            # -- Detect keypoints
            keypoints, _ = blob_detect(frame, blue_min, blue_max, blur=3,
                                       blob_params=None, search_window=window, imshow=False)
            # -- Draw search window
            frame = draw_window(frame, window)
            # -- click ENTER on the image window to proceed
           # center = get_blob_relative_position(frame,keypoints)
            if(not keypoints):
                message = str(int(999)) + '\n'
                message1 = str(int(999)) + '\n'
               # message2 = str(int(999)) + '\n'
                ser.write(message.encode())
                ser.write(message1.encode())
                # print("Find the ball ", '\n')
                # print("x-axis: "+ message)
                # print("Y-axis: " + message1)
            old=None
            if (keypoints):
                old=keypoints[0]
                # print(old)
                # print(keypoints)

            for i, keyPoint in enumerate(keypoints):

                oldx, oldy=get_blob_relative_position(frame,old)
                newx, newy = get_blob_relative_position(frame, keyPoint)
                oldr=(oldx*oldx+oldy*oldy)**0.5
                newr = (newx + newy *newy)*0.5
                # print(oldr)
                # print(newr)
                if (oldr > newr):
                    old=keyPoint


            if (old):
                oldx, oldy = get_blob_relative_position(frame, old)
                cv_keypoints =[]
                cv_keypoints.append(old)
                draw_keypoints(frame,cv_keypoints,imshow=True)
                x, y=get_blob_relative_position(frame,old)
                s=old.size
                oldr = (oldx * oldx + oldy * oldy) ** 0.5
                x2=x*10*10
                y2=y*10*10
                # print(s)
                #print(x2)
                     # Format the message as "X,Y\n)
                  # To make the robot take its time to move

                message = str(int(x2)) + '\n'
                message1=str(int(oldr*100)) + '\n'
                message2 = str(int(s)) + '\n'
                ser.write(message.encode())
                ser.write(message2.encode())
                 # ser.write(message2.encode())

                print("X-axis:",message)
                # print("Y-axis:", message1)
                print("distace:", message2)
            else:
                cv_keypoints=[]
                draw_keypoints(frame, cv_keypoints, imshow=True)

            # -- press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    else:
        # -- Read image list from file:
        image_list = []
        image_list.append(cv2.imread("kora.jpg"))
        # image_list.append(cv2.imread("blob2.jpg"))
        # image_list.append(cv2.imread("blob3.jpg"))

        for image in image_list:
            # -- Detect keypoints
            keypoints, _ = blob_detect(image, blue_min, blue_max, blur=5,
                                       blob_params=None, search_window=window, imshow=True)

            image = blur_outside(image, blur=15, window_adim=window)
            cv2.imshow("Outside Blur", image)
            cv2.waitKey(0)

            image = draw_window(image, window, imshow=True)
            # -- enter to proceed
            cv2.waitKey(0)

            # -- click ENTER on the image window to proceed
            image = draw_keypoints(image, keypoints, imshow=True)
            cv2.waitKey(0)
            # -- Draw search window

            image = draw_frame(image)
            cv2.imshow("Frame", image)
            cv2.waitKey(0)