# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):

   # Write histogram equalization here
   #########################################################################
   # START CUSTOM CODE
   #########################################################################

   img_split = cv2.split(img_in)  # Split image into color channels
   equalized = []                 # List of equalized image matrices for each channel

   # Apply histogram equalization to each color channel
   for channel in img_split:

      histogram = np.histogram(channel, 256)[0]
      cdf = np.cumsum(histogram, dtype=float)

      # Convert CDF bins to percentages and map to pixel value, as shown in Prince
      cdf = np.round((cdf / cdf.max()) * 255)

      # Use channel pixel values as keys into cdf values
      eq = cdf[channel].astype('uint8')

      # Store result of current channel
      equalized.append(eq)
   
   # Merge equalized channels
   img_in = cv2.merge(equalized)

   #########################################################################
   # END CUSTOM CODE
   #########################################################################
   img_out = img_in # Histogram equalization result 
   
   return True, img_out
   
def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
   
   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = sys.argv[3] + "output1.png"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):
	
   # Write low pass filter here
   #########################################################################
   # START CUSTOM CODE
   #########################################################################

   img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

   # Apply FT and shift to center
   ft_shift = np.fft.fftshift(np.fft.fft2(img_in))

   # Make mask of Zero matrix with only 20x20 center having value 1
   rows, cols = img_in.shape
   mask = np.zeros((rows, cols), np.uint8)
   mask[rows/2 - 10 : rows/2 + 10, cols/2 - 10 : cols/2 + 10] = 1

   # Apply the mask via pointwise multiplication
   ft_shift = ft_shift * mask

   # Invert the modified FT to implement high pass filtering
   img_in = np.abs(np.fft.ifft2(np.fft.ifftshift(ft_shift)))

   #########################################################################
   # END CUSTOM CODE
   #########################################################################
   img_out = img_in # Low pass filter result
	
   return True, img_out

def high_pass_filter(img_in):

   # Write high pass filter here
   #########################################################################
   # START CUSTOM CODE
   #########################################################################

   img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
   
   # Apply FT and shift to center
   ft_shift = np.fft.fftshift(np.fft.fft2(img_in))

   # Set 20x20 center of shifted FT to 0
   rows, cols = img_in.shape
   ft_shift[rows/2 - 10 : rows/2 + 10, cols/2 - 10 : cols/2 + 10] = 0

   # Invert the modified FT to implement high pass filtering
   img_in = np.abs(np.fft.ifft2(np.fft.ifftshift(ft_shift)))

   #########################################################################
   # END CUSTOM CODE
   #########################################################################
   img_out = img_in # High pass filter result
   
   return True, img_out
   
def deconvolution(img_in):
   
   # Write deconvolution codes here
   #########################################################################
   # START CUSTOM CODE
   #########################################################################

   img_ft = np.fft.fftshift(np.fft.fft2(np.float32(img_in)))

   # Get Gaussian kernel and map to Fourier Domain
   gk = cv2.getGaussianKernel(21, 5)
   gk = gk * gk.T
   gk_ft = np.fft.fftshift(np.fft.fft2(np.float32(gk), (img_in.shape[0], img_in.shape[1])))

   # Use FT rules to reconstruct original image
   original_img = img_ft / gk_ft
   img_in = np.abs(np.fft.ifft2(np.fft.ifftshift(original_img)))

   img_in = cv2.convertScaleAbs(img_in, alpha=255.0)

   #########################################################################
   # END CUSTOM CODE
   #########################################################################
   img_out = img_in # Deconvolution result
   
   return True, img_out

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "output2LPF.png"
   output_name2 = sys.argv[4] + "output2HPF.png"
   output_name3 = sys.argv[4] + "output2deconv.png"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):

   # Write laplacian pyramid blending codes here
   #########################################################################
   # START CUSTOM CODE
   #########################################################################

   # Resize images to have the same dimensions
   img_in1 = img_in1[:, :img_in1.shape[0]]
   img_in2 = img_in2[:img_in1.shape[0], :img_in1.shape[0]]

   # Generate Gaussian pyramids for both images
   g_1 = img_in1.copy()  # Used to recurse on image 1
   g_2 = img_in2.copy()  # Used to recurse on image 2
   gp_1 = [g_1]          # Holds downsized images from image 1
   gp_2 = [g_2]          # Holds downsized images from image 2

   for i in xrange(6):
      g_1 = cv2.pyrDown(g_1)
      g_2 = cv2.pyrDown(g_2)
      gp_1.append(g_1)
      gp_2.append(g_2)

   # Generate Laplacian pyramids for both images
   lp_1 = [gp_1[5]]  # Holds Laplacian images (first element is smallest Gaussian)
   lp_2 = [gp_2[5]]  # Holds Laplacian images (first element is smallest Gaussian)

   for i in xrange(5, 0, -1):

      # Expand downsized image
      expanded_1 = cv2.pyrUp(gp_1[i])
      expanded_2 = cv2.pyrUp(gp_2[i])

      # Compute Laplacian image and insert into list
      lp_1.append(cv2.subtract(gp_1[i - 1], expanded_1))
      lp_2.append(cv2.subtract(gp_2[i - 1], expanded_2))

   # Concatenate left and right halfs of Laplacians
   lp_concat = []  # Holds concatenated images

   for l_1, l_2 in zip(lp_1, lp_2):

      # Calculate splitting point
      rows, cols, dpt = l_1.shape
      split = cols/2

      # Concat and store
      concat = np.hstack((l_1[:, 0:split], l_2[:, split:]))
      lp_concat.append(concat)

   # Reconstruct blended image
   blend = lp_concat[0]

   for i in xrange(1, 6):
      blend = cv2.pyrUp(blend)
      blend = cv2.add(blend, lp_concat[i])

   img_in1 = blend

   #########################################################################
   # END CUSTOM CODE
   #########################################################################
   img_out = img_in1 # Blending result
   
   return True, img_out

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)
   
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "output3.png"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
	  
      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
	 sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
	 print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
