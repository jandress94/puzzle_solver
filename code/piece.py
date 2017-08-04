import sys
from PIL import Image
import numpy as np

from image_utils import *

class PuzzlePiece:
    def __init__(self, image_mat):
        if areAllNotPiecePixels(image_mat):
            sys.exit("Got a puzzle piece with no pixels")

        image_mat = defrag(image_mat)
        image_mat = crop(image_mat)
        image_mat = straighten(image_mat)
        image_mat = crop(image_mat)

        image_mat = getPieceCorners(image_mat)

        self.image_mat = image_mat

    def saveImage(self, filename):
        save_mat = np.copy(self.image_mat)

        for x in xrange(save_mat.shape[0]):
            for y in xrange(save_mat.shape[1]):
                if not areAllPiecePixels(save_mat[x, y]):
                    save_mat[x, y, :] = np.array([255, 0, 0])

        Image.fromarray(save_mat.astype('uint8')).save(filename)

    def countPix(self):
        count = 0
        for x in xrange(self.image_mat.shape[0]):
            for y in xrange(self.image_mat.shape[1]):
                if not areAllPiecePixels(save_mat[x,y]):
                    continue
                count += 1
        return count