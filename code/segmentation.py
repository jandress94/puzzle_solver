import numpy as np
import sys

from piece import *
from constants import *

class Segmenter:
    def __init__(self):
        pass

    def extract(self, image_mat):
        sys.exit('A segmenter must implement an "extract" method')

class SolidBackSinglePieceSegmenter(Segmenter):
    def __init__(self):
        Segmenter.__init__(self)

    def extract(self, image_mat):
        if not allCornersSame(image_mat):
            sys.exit('Got an image which had different values in the corners')
        
        w, h, _ = image_mat.shape

        corner_color = np.copy(image_mat[0, 0, :])

        pixel_queue = [(0,0)]
        visited = np.zeros((w, h))

        while len(pixel_queue) > 0:
            x, y = pixel_queue.pop(0)

            if not isInBounds(image_mat, x, y):
                continue

            if visited[x, y] != 0:
                continue

            visited[x, y] = 1

            if np.max(np.abs(image_mat[x, y, :] - corner_color)) > SEG_THRESHOLD:
                continue

            image_mat[x, y] = NON_PIECE_VAL
            pixel_queue.append((x, y-1))
            pixel_queue.append((x, y+1))
            pixel_queue.append((x-1, y))
            pixel_queue.append((x+1, y))

        if (image_mat < 0).all():
            return []

        return [PuzzlePiece(image_mat)]

def allCornersSame(image_mat):
    w, h, _ = image_mat.shape
    
    v1 = image_mat[0, 0, :]
    v2 = image_mat[0, h - 1, :]
    v3 = image_mat[w - 1, 0, :]
    v4 = image_mat[w - 1, h - 1, :]

    return np.array_equal(v1, v2) and np.array_equal(v2, v3) and np.array_equal(v3, v4)