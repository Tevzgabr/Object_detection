import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interpn
import PIL.Image as im


# calibrationFile = "logitech9000.yml"
# calibrationParams = cv.FileStorage(calibrationFile, cv.FILE_STORAGE_READ)
# camera_matrix = calibrationParams.getNode("camera_matrix").mat()
# dist_coeffs = calibrationParams.getNode("distortion_coefficients").mat()
def img_cal(calib_img, test_img):

	def showImage(iImage, iTitle = ""):
	    """
	    Prikaz slike s matplotlib
	    
	    Vhod; slika (sivinska/barvna), naslov
	    
	    ---------------------------------------------------------------------------
	    
	    return /
	    """
	    plt.figure()
	    plt.imshow(iImage, cmap = cm.Greys_r)
	    plt.suptitle(iTitle)
	    plt.xlabel("x")
	    plt.ylabel("y")
	    plt.axes().set_aspect("equal","datalim")
	    plt.show()

	def geomCalibImage(iPar, iImage, iCoorX, iCoorY):
	    iCoorU, iCoorV = geomCalibTrans(iPar, iCoorX, iCoorY)  # naredimo kalibracijsko preslikavo
	    oImage = np.zeros_like(iCoorX)
	    dy, dx = iImage.shape
	    points = np.array([np.arange(dy), np.arange(dx)])
	    oImage = interpn(points, iImage, (iCoorV[::1, ::1], iCoorU[::1, ::1]), method = "linear", bounds_error = False)
	    return oImage

	def transProjective2D( iPar, iCoorX, iCoorY ):
	    # preveri vhodne podatke
	    iPar = np.asarray( iPar )
	    iCoorX = np.asarray( iCoorX )
	    iCoorY = np.asarray( iCoorY )
	    if np.size(iCoorX) != np.size(iCoorY):
	        print('Stevilo X in Y koordinat je razlicno!')  
	    # izvedi projektivno preslikavo
	    oDenom = iPar[6] * iCoorX + iPar[7] * iCoorY + 1
	    oCoorU = iPar[0] * iCoorX + iPar[1] * iCoorY + iPar[2]
	    oCoorV = iPar[3] * iCoorX + iPar[4] * iCoorY + iPar[5]
	    # vrni preslikane tocke
	    return oCoorU/oDenom, oCoorV/oDenom

	def transRadial( iK, iUc, iVc, iCoorU, iCoorV ):
	    """Funkcija za preslikavo z Brownovim modelom distorzij"""    
	    # preveri vhodne podatke
	    iK = np.array( iK )
	    iCoorU = np.array( iCoorU )
	    iCoorV = np.array( iCoorV )
	    if np.size(iCoorU) != np.size(iCoorV):
	        print('Stevilo U in V koordinat je razlicno!')  
	    # odstej koodinate centra
	    oCoorUd = iCoorU - iUc; oCoorVd = iCoorV - iVc
	    # pripravi izhodne koordinate
	    sUd = np.max( np.abs( oCoorUd ) )
	    sVd = np.max( np.abs( oCoorVd ) )    
	    oCoorUd = oCoorUd / sUd
	    oCoorVd = oCoorVd / sVd
	    # preracunaj radialna popacenja
	    R2 = oCoorUd**2.0 + oCoorVd**2.0    
	    iK = iK.flatten()
	    oCoorRd = np.ones_like( oCoorUd )
	    for i in range(iK.size):
	        oCoorRd = oCoorRd + iK[i]*(R2**(i+1))
	    # izracunaj izhodne koordinate
	    oCoorUd = oCoorUd * oCoorRd * sUd + iUc
	    oCoorVd = oCoorVd * oCoorRd * sVd + iVc
	    return oCoorUd, oCoorVd

	def geomCalibTrans(iPar, iCoorX, iCoorY):
	    iParProj = iPar[:8]
	    iParRad = iPar[8:]
	    iCoorUt, iCoorVt = transProjective2D(iParProj, iCoorX, iCoorY)
	    iCoorUt, iCoorCt = transRadial(iParRad[2:], iParRad[0], iParRad[1], iCoorUt, iCoorVt)
	    return iCoorUt, iCoorVt

	def geomCalibErr(iPar, iCoorU, iCoorV, iCoorX, iCoorY):
	    iCoorUt, iCoorVt = geomCalibTrans(iPar, iCoorX, iCoorY)
	    oErr = np.mean((iCoorU - iCoorUt)**2 + (iCoorV - iCoorVt)**2)
	    return oErr

	def mapAffineApprox2D(iPtsRef, iPtsMov):
	    iPtsRef = np.matrix(iPtsRef)
	    iPtsMov = np.matrix(iPtsMov)
	    iPtsRef = addHomCoord2D(iPtsRef)
	    iPtsMov = addHomCoord2D(iPtsMov)
	    iPtsRef = iPtsRef.transpose()
	    iPtsMov = iPtsMov.transpose()
	    oMat2D = iPtsRef * iPtsMov.transpose() * np.linalg.inv(iPtsMov * iPtsMov.transpose())
	    return oMat2D

	def addHomCoord2D(iPts): 
	    if iPts.shape[-1] == 3: 
	        return iPts
	    iPts = np.hstack((iPts, np.ones((iPts.shape[0],1)))) 
	    return iPts
	#%% Koda kalibracija slike

	def loadImage(iPath):
	    oImage = np.array(im.open(iPath))
	    return oImage


	from scipy.optimize import fmin



	iCalImage = calib_img
	iCoorX = np.array([20,100,180,180,180,100,20,20])
	iCoorY = np.array([40,40,40,100,160,160,160,100])  # points in the 3D space based on the chess board


	
	#  manually defined points on the image (the same points on the chess board as in the real world)
	pts = np.array([[12,21],[56,21],[100,20],[102,53],[101,86],[57,86],[13,87],[12,54]]) 

	iCoorU = pts[:,0].flatten()
	iCoorV = pts[:,1].flatten()
	plt.plot(iCoorU, iCoorV, "or", markersize = 5)

	# # Initiall approximation for the projective and radial mapping
	ptsUV = np.vstack((iCoorU, iCoorV)).transpose()
	ptsXY = np.vstack((iCoorX, iCoorY)).transpose()
	oMat = mapAffineApprox2D(ptsUV, ptsXY)

	Uc = iCalImage.shape[1]/2
	Vc = iCalImage.shape[0]/2
	iParAffine = np.array([oMat[0,0], oMat[0,1], oMat[0,2], oMat[1,0], oMat[1,1], oMat[1,2], 0, 0, Uc, Vc, 0])
	oErrAffine = geomCalibErr(iParAffine, iCoorU, iCoorV, iCoorX, iCoorY) # napaka afine preslikave

	F = lambda x: geomCalibErr(x,iCoorU, iCoorV, iCoorX, iCoorY)
	iParOpt = fmin(func = F, x0 = iParAffine, maxiter = 4100, xtol = 1e-8, ftol = 1e-8, disp = 1)


	# #Function that, for the given optimal paramteres from the camera callibration normalizes the image in the matric space
	iCoorX, iCoorY = np.meshgrid(np.arange(0, 1100, 1), np.arange(20, 900, 1),sparse=False, indexing='xy')
	Opt_calibImage = np.zeros([880,1100,3])
	
	    # Geometrical alignment with the optimal parameters
	for i in range(3):
		Opt_calibImage[:,:,i] = geomCalibImage(iParOpt, test_img[:,:,i], iCoorX, iCoorY) 
	Opt_calibImage = Opt_calibImage.astype(np.uint8)
	    
	image = Opt_calibImage

	return(image)