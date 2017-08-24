import sys
from PIL import Image
import numpy as np

from image_utils import *

class PieceEdge:
    def __init__(self, piece, path, out_score, index):
        self.piece = piece
        self.path = path
        self.index = index
        if abs(out_score) < EDGE_STICK_OUT_SCORE_CUTOFF:
            self.piece_type = 0
        elif out_score > 0:
            self.piece_type = 1
        else:
            self.piece_type = -1

    def isStickingOut(self):
        return self.piece_type == 1

    def isStickingIn(self):
        return self.piece_type == -1

    def isFlat(self):
        return self.piece_type == 0

class PuzzlePiece:
    def __init__(self, image_mat):
        if areAllNotPiecePixels(image_mat):
            sys.exit("Got a puzzle piece with no pixels")

        image_mat = defrag(image_mat)
        image_mat = crop(image_mat)
        image_mat = straighten(image_mat)
        image_mat = crop(image_mat)
        self.image_mat = image_mat

        edge_paths, edge_out_scores = getEdgeTypes(image_mat)
        self.edges = [PieceEdge(self, edge_paths[i], edge_out_scores[i], i) for i in xrange(len(edge_paths))]

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