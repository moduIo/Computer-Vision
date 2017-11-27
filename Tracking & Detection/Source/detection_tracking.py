import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")

def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist

def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

############################################
# START CUSTOM CODE
############################################

#
# Camshift implementation
#
def camshift_tracker(v, file_name):

    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0

    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt = (c + w/2, r + h/2)
    output.write("%d,%d,%d\n" % (frameCounter, pt[0], pt[1])) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the normalized HSV histogram in the window
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        # Convert frame into histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Find midpoint of window point
        c,r,w,h = track_window
        pt = (c + w/2, r + h/2)

        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter, pt[0], pt[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()

#
# Particle Filter implementation
#
def particle_tracker(v, file_name):

    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0

    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt = (c + w/2, r + h/2)
    output.write("%d,%d,%d\n" % (frameCounter, pt[0], pt[1])) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    roi = hsv_histogram_for_window(frame, (c,r,w,h))

    # initialize the tracker
    n_particles = 200

    init_pos = np.array([c + w/2, r + h/2], int)           # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos  # Init particles to init position
    weights = np.ones(n_particles) / n_particles           # Set weights uniformly (at first)

    while(1):
        ret, frame = v.read() # read another frame
        if ret == False:
            break

        # Particle motion model: uniform step (TODO: find a better motion model)
        stepsize = 22.5
        im_h, im_w, im_c = frame.shape
        np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

        # Clip out-of-bounds particles
        particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)

        # Get backprojection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist_bp = cv2.calcBackProject([hsv_frame], [0], roi, [0,180], 1)

        f = particleevaluator(hist_bp, particles.T) # Evaluate particles
        weights = np.float32(f.clip(1))             # Weight ~ histogram response
        weights /= np.sum(weights)                  # Normalize w
        pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

        # Display tracking
        #frame = cv2.circle(frame, (int(pos[0]), int(pos[1])), 2, (0, 255, 0), -1)
        #cv2.imshow('image',frame)
        #cv2.waitKey(0)
        
        if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
            particles = particles[resample(weights),:] # Resample particles according to weights

        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter, pos[0], pos[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()

#
# Kalman Filter implementation
#
def kalman_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0

    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt = (c + w/2, r + h/2)
    output.write("%d,%d,%d\n" % (frameCounter, pt[0], pt[1])) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # initialize the tracker
    kalman = cv2.KalmanFilter(4,2,0)
    state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                        [0., 1., 0., .1],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state

    while(1):
        ret,frame = v.read() # read another frame
        if ret == False:
            break

        # Predict location
        prediction = kalman.predict()

        # Try to detect face
        c,r,w,h = detect_one_face(frame)
        track_window = (c,r,w,h)

        if track_window != (0, 0, 0, 0):
            posterior = kalman.correct(np.array([c+w/2, r+h/2], dtype='float64'))
            pt = (posterior[0], posterior[1])
            frame = cv2.circle(frame, (int(posterior[0]), int(posterior[1])), 2, (0, 255, 0), -1)

        else:
            pt = (prediction[0], prediction[1])
            frame = cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 2, (0, 255, 0), -1) 

        frame = cv2.rectangle(frame, (c, r), (c + w, r + h), 255, 2)

        # Display tracking
        #cv2.imshow('image',frame)
        #cv2.waitKey(0)

        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter, pt[0], pt[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()

############################################
# END CUSTOM CODE
############################################

if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        camshift_tracker(video, "output_camshift.txt")
    elif (question_number == 2):
        particle_tracker(video, "output_particle.txt")
    elif (question_number == 3):
        kalman_tracker(video, "output_kalman.txt")
    elif (question_number == 4):
        of_tracker(video, "output_of.txt")