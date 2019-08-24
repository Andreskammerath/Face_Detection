# face verification with the VGGFace2 model
import matplotlib
import os
from matplotlib import patches
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

#function that detects faces
def recog_faces(img_folder,image,save_in_folder):
	#loads the image an array of pixels directly
	pixels = pyplot.imread(img_folder + image)
	#create the detector, using default weights
	detector = MTCNN()
	#actual detection
	results = detector.detect_faces(pixels)
	fig,ax = pyplot.subplots(1)
	#extract the bounding box from every face
	for i in range(len(results)):
		x1,y1,width,height = results[i]['box']
		rect = patches.Rectangle((x1,y1),width,height,linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
	ax.imshow(pixels)
	if not os.path.isdir(save_in_folder):
		os.mkdir(save_in_folder)
	else:
		pass
	pyplot.savefig(save_in_folder+"/"+ image,bbox_inches='tight')


OUTPUT_FOLDER = "output_images"

IMG_FOLDER_PATH = "images/"

images = [f for f in os.listdir(IMG_FOLDER_PATH)]

#Perform detection for every img
print(images)
for img in images:
	recog_faces(IMG_FOLDER_PATH,img,OUTPUT_FOLDER)
