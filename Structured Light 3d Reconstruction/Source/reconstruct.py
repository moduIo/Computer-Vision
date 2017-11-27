# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================
import sys
import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_avg   = (ref_white + ref_black) / 2.0
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region

    h,w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h,w), dtype=np.uint16)

    # Get RGB values of image
    ref_color = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_COLOR) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    
    colors = np.zeros((h,w,3), dtype=np.float32)
    for x in range(w):
        for y in range(h):
            if proj_mask[y, x]:
                colors[y, x, :] = ref_color[y, x, :]

    # analyze the binary patterns from the camera
    for i in range(0,15):
        # read the file
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2), cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask

        # this code corresponds with the binary pattern code
        bit_code = 1 << i

        # TODO: populate scan_bits by putting the bit_code according to on_mask
        scan_bits = np.add(scan_bits, on_mask * bit_code)

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    camera_points = []
    projector_points = []
    rgb_points = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code

            [x_p, y_p] = binary_codes_ids_codebook[scan_bits[y,x]]
            if x_p >= 1279 or y_p >= 799: # filter
                continue
                
            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points
            camera_points.append([x/2.0, y/2.0])
            projector_points.append([x_p, y_p])
            rgb_points.append([colors[y, x, 2] * 255, colors[y, x, 1] * 255, colors[y, x, 0] * 255])

            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2

    # Construct correspondence image
    correspondence_image = np.zeros((h, w, 3), dtype=np.float32)

    for i in range(len(camera_points)):
        correspondence_image[int(camera_points[i][1] * 2.0), int(camera_points[i][0] * 2.0)] = (0, projector_points[i][1]/800.0, projector_points[i][0]/1280.0)

    print("saving correspondence.jpg")
    plt.imshow(correspondence_image[:,:,::-1])
    plt.savefig('correspondence.jpg')
    #plt.show()

    # now that we have 2D-2D correspondences, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
    
    # Get projection matrices
    camera_homogeneous = np.matmul(camera_K, np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
    projector_homogeneous = np.matmul(projector_K, np.concatenate((projector_R, projector_t), axis=1))

    undistorted_camera = cv2.undistortPoints(np.array([camera_points], dtype=np.float32), camera_K, camera_d, P=camera_K)
    undistorted_projector = cv2.undistortPoints(np.array([projector_points], dtype=np.float32), projector_K, projector_d, P=projector_K)

    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    homogeneous_3d = cv2.triangulatePoints(projector_homogeneous, camera_homogeneous, np.transpose(undistorted_projector[0]), np.transpose(undistorted_camera[0]))

    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
    points_3d = cv2.convertPointsFromHomogeneous(np.transpose(homogeneous_3d))

    # TODO: name the resulted 3D points as "points_3d"
    # after cv2.triangulatePoints and cv2.convertPointsFromHomogeneous
    # apply another filter on the Z-component
    mask = (points_3d[:,:,2] > 200) & (points_3d[:,:,2] < 1400)

    # Create color 3d point cloud
    d1, d2, d3 = points_3d.shape
    points_3d_rgb = np.zeros((d1, d2, 6))

    for i in range(len(points_3d)):
        points_3d_rgb[i][0] = np.concatenate((points_3d[i][0], rgb_points[i]))
    
    return points_3d[mask[:,0],:,:], points_3d_rgb[mask[:,0],:,:]
	
def write_3d_points(points_3d):
	
	# ===== DO NOT CHANGE THIS FUNCTION =====
	
    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d\n"%(p[0,0],p[0,1],p[0,2]))
    
def write_3d_color_points(rgb_points):
    
    print("write output rgb point cloud")
    print(rgb_points.shape)
    output_name = sys.argv[1] + "output_color.xyz"
    with open(output_name,"w") as f:
        for p in rgb_points:
            f.write("%d %d %d %d %d %d\n"%(p[0,0],p[0,1],p[0,2],p[0,3],p[0,4],p[0,5]))

if __name__ == '__main__':

	# ===== DO NOT CHANGE THIS FUNCTION =====
	
	# validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d, rgb_points = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
    write_3d_color_points(rgb_points)