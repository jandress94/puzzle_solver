import os
import sys
from PIL import Image
import numpy as np
import random
import scipy.ndimage

from constants import *
from segmentation import *

pieces_folder_name = os.path.join(DATA_FOLDER_NAME, 'pieces')
if not (os.path.exists(pieces_folder_name) and os.path.isdir(pieces_folder_name)):
    sys.exit("Couldn't find the pieces folder: %s" % (pieces_folder_name))

pieces_list = []

segmenter = SolidBackSinglePieceSegmenter()

for piece_filename in os.listdir(pieces_folder_name):
    if not piece_filename.endswith('.jpg'): continue
    if DEBUG: print("processing piece %s" % (piece_filename))

    image_mat = Image.open(os.path.join(pieces_folder_name, piece_filename))
    # print(np.array(image_mat).dtype)
    image_mat = np.array(image_mat).astype('int16')
    # image_mat = cv2.imread(os.path.join(pieces_folder_name, piece_filename)).astype('int16')

    # image_mat = np.rot90(image_mat, random.randint(0, 3))
    rand_rot = random.randint(0, 360)
    # rand_rot = 344
    print("rotating %d" % (rand_rot))
    image_mat = scipy.ndimage.interpolation.rotate(image_mat, rand_rot, order=0, cval=255)

    pieces_list += segmenter.extract(image_mat)

if DEBUG: print("%d pieces" % (len(pieces_list)))

for i in xrange(len(pieces_list)):
    pieces_list[i].saveImage(os.path.join(TEMP_FOLDER_NAME, 'piece%d.jpg' % (i)))

out_edges = []
in_edges = []
flat_edges = []
for i in xrange(len(pieces_list)):
    for edge in pieces_list[i].edges:
        if edge.isStickingOut():
            out_edges.append(edge)
        elif edge.isStickingIn():
            in_edges.append(edge)
        else:
            flat_edges.append(edge)
