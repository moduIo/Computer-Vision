# Computer-Vision
CSE 527 Computer Vision - Fall 2017 - Stony Brook University

All implementations are written in Python 2.7 using OpenCV.

## Gaussian Pyramids & Blending 
Contains an implementation of kernel deconvolution in the frequency domain as well as Laplacian Pyramid blending of two images.

## Image Stitching & Panoramas
Uses RANSAC to fit homography and affine transformations for image matching and panoramic stitching.

## Tracking & Detection
Implements a human face detector and tracker which detects faces using Viola-Jones.  Tracking is accomplished using Camshift, Kalman filter, and particle filtering.

## Segmentation
Uses SLIC to segment an image using superpixels, also includes an interactive foreground/background segmentation implementation which will segment based on user's input markings.

## Structured Light 3d Reconstruction
Reconstructs a sparse 3d point cloud of an image processed via structured light.
