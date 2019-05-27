# Object_detection
In this repositorie there are 4 python codes, where 3 there are only functions which are to be called in the main function (koncna_vid).
we also included the Logitech 9000 camera calibration file (logitech.yml), which is used for the camera calibration, when the ArUco marker is being detected. 
The code koncna_vid can be used with the included pictures, which were taken with the Kuka robot, for which this code was designed. 
The code first calibrates the test image with the calibration image, and then all the image procesing is done on the calibrated image, the image is then masked so that only the objects are computed. 
