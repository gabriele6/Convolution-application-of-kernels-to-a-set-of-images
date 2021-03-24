import numpy as np
import cv2
from skimage.exposure import rescale_intensity
import glob
import os


def convolution(image, kernel, rescale=False, o_range=(-255, 255)):
	#get input sizes
	imgRows, imgCols = image.shape[0], image.shape[1]
	kerRows, kerCols = kernel.shape[0], kernel.shape[1]

	#create empty output matrix
	outRows = imgRows + kerRows - 1
	outCols = imgCols + kerCols - 1
	out = np.zeros((outRows, outCols))
    
	#slide over every output cell to calculate its value
	for currentOutRow in range(outRows):
		for currentOutCol in range(outCols):
			#slide over every kernel cell to calculate the output value of the current cell
			for currentKerRow in range(kerRows):
				for currentKerCol in range(kerCols):
					rowDiff = currentOutRow - currentKerRow #<---sliding bottom to top
					colDiff = currentOutCol - currentKerCol #<---sliding right to left
					if (rowDiff >= 0) and (rowDiff < imgRows):
						if(colDiff >= 0) and (colDiff < imgCols):
							oldOut = out[currentOutRow, currentOutCol]
							currentKer = kernel[currentKerRow, currentKerCol]
							currentImg = image[rowDiff, colDiff]
							out[currentOutRow, currentOutCol] = oldOut + (currentKer * currentImg)
	if(rescale):
		out = rescale_intensity(out, out_range=o_range)
	return out


def getKernel(filterName):
	if (filterName == "sobel"):
		filterX = np.array(([-1, 0, 1],[-2, 0, 2],[-1, 0, 1]), dtype="int")
		filterY = np.array(([1, 2, 1],[0, 0, 0],[-1, -2, -1]), dtype="int")
	elif (filterName == "prewitt"):
		filterX = np.array(([-1, 0, 1],[-1, 0, 1],[-1, 0, 1]), dtype="int")
		filterY = np.array(([1, 1, 1],[0, 0, 0],[-1, -1, -1]), dtype="int")
	elif (filterName == "roberts"):
		filterX = np.array(([+1, 0],[0, -1]), dtype="int")
		filterY = np.array(([0, -1],[1, 0]), dtype="int")
	elif (filterName == "blur"):
		filterX = np.array(([0.0625, 0.125, 0.0625],[0.125, 0.25, 0.125],[0.0625, 0.125, 0.0625]), dtype="float")
		filterY = np.array(([0, 0, 0],[0, 0, 0],[0, 0, 0]), dtype="int")
	elif (filterName == "sharpen"):
		filterX = np.array(([0, -1, 0],[-1, 5, -1],[0, -1, 0]), dtype="int")
		filterY = np.array(([0, 0, 0],[0, 0, 0],[0, 0, 0]), dtype="int")
	elif (filterName == "outline"):
		filterX = np.array(([-1, -1, -1],[-1, 8, -1],[-1, -1, -1]), dtype="int")
		filterY = np.array(([0, 0, 0],[0, 0, 0],[0, 0, 0]), dtype="int")
	return filterX, filterY


def saveImage(folder, imgName, filterName, Gx, Gy, magnitude):
	cv2.imwrite("./" + folder + "/" + imgName + "_" + filterName + "_Gx.jpg",Gx)
	cv2.imwrite("./" + folder + "/" + imgName + "_" + filterName + "_Gy.jpg",Gy)
	cv2.imwrite("./" + folder + "/" + imgName + "_" + filterName + "_magnitude.jpg",np.float32(magnitude))


def computeImage(img, filterX, filterY):
	Gx = convolution(img, filterX)
	Gy = convolution(img, filterY)
	magnitude = np.hypot(Gx, Gy)
	return Gx, Gy, magnitude


#grabbing pictures in the folder
folder = "pics/"
images = [os.path.basename(x) for x in glob.glob(folder + '*.bmp')]
images = [os.path.splitext(x)[0] for x in images]

for image in images:
	print("Elaborating " + image + ".bmp")
	img = cv2.cvtColor(cv2.imread("./" + folder + image + ".bmp"), cv2.COLOR_BGR2GRAY)
	cv2.imwrite("./out" + "/" + image + "_original" + ".jpg",img)


	#apply filters and save images
	filterX, filterY = getKernel("sobel")
	Gx, Gy, magnitude = computeImage(img, filterX, filterY)
	saveImage("out", image, "sobel", Gx, Gy, magnitude)
    
	filterX, filterY = getKernel("prewitt")
	Gx, Gy, magnitude = computeImage(img, filterX, filterY)
	saveImage("out", image, "prewitt", Gx, Gy, magnitude)

	filterX, filterY = getKernel("roberts")
	Gx, Gy, magnitude = computeImage(img, filterX, filterY)
	saveImage("out", image, "roberts", Gx, Gy, magnitude)
    
	#BONUS: blur, sharpen, outline, blur&outline filters
	filterX, _ = getKernel("blur")
	Gx = convolution(img, filterX)
	cv2.imwrite("./out" + "/" + image + "_blur" + ".jpg",Gx)
	filterX, _ = getKernel("sharpen")
	Gx = convolution(img, filterX)
	cv2.imwrite("./out" + "/" + image + "_sharpen" + ".jpg",Gx)
	filterX, _ = getKernel("outline")
	Gx = convolution(img, filterX)
	cv2.imwrite("./out" + "/" + image + "_outline" + ".jpg",Gx)
	filterY, _ = getKernel("blur")
	Gx = convolution(convolution(img, filterY),filterX)
	cv2.imwrite("./out" + "/" + image + "_blur_outline" + ".jpg",Gx)

print("Convolution: Done!")
