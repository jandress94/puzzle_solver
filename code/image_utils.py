import numpy as np
import random
import math
import scipy.ndimage

from constants import *

def isInBounds(img, x, y):
    return x >= 0 and y >= 0 and x < img.shape[0] and y < img.shape[1]

def isOnBorder(img, x, y):
    return x == 0 or y == 0 or x == img.shape[0] - 1 or y == img.shape[1] - 1

def getNeighbors(img, x, y, include_diags = True):
    neighbor_list = []
    for dx in xrange(-1, 2):
        for dy in xrange(-1, 2):
            if dx == 0 and dy == 0:
                continue
            if not isInBounds(img, x + dx, y + dy):
                continue
            if not (include_diags or abs(dx) + abs(dy) == 1):
                continue

            neighbor_list.append((x + dx, y + dy))
    return neighbor_list

def areAllPiecePixels(img):
    return (img >= 0).all()

def areAllNotPiecePixels(img):
    return (img < 0).all()

def getInteriorPoint(img):
    w, h, _ = img.shape

    # if the center point is interior, return that
    if areAllPiecePixels(img[w/2, h/2]):
        return w/2, h/2

    # otherwise, pick randomly until one is found
    while True:
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        if areAllPiecePixels(img[x, y]):
            return x, y


def crop(img):
    # cut off rows / cols from the border until you get to a row / col that has a real pixel
    while True:
        if areAllNotPiecePixels(img[0, :, :]):
            img = img[1:, :, :]
        else:
            break;

    while True:
        if areAllNotPiecePixels(img[-1, :, :]):
            img = img[:-1, :, :]
        else:
            break;

    while True:
        if areAllNotPiecePixels(img[:, 0, :]):
            img = img[:, 1:, :]
        else:
            break;

    while True:
        if areAllNotPiecePixels(img[:, -1, :]):
            img = img[:, :-1, :]
        else:
            break;

    return img

def defrag(img):
    w, h, _ = img.shape

    visited = np.zeros((w, h))
    conn_comps = []

    # loop over the image to find all of the connected components
    for x in xrange(w):
        for y in xrange(h):
            if visited[x, y] != 0:
                continue

            if areAllPiecePixels(img[x, y]):
                component = set()
                # now that we found an unvisted real pixel, find all the pixels connected to it
                defrag_bfs(img, x, y, visited, component)
                conn_comps.append(component)

            visited[x, y] = 1

    # find the largest connected component
    max_comp_size = -1
    max_comp_ind = -1
    for i in xrange(len(conn_comps)):
        if len(conn_comps[i]) > max_comp_size:
            max_comp_size = len(conn_comps[i])
            max_comp_ind = i

    # clear out all but the largest connected component
    del conn_comps[max_comp_ind]
    for component in conn_comps:
        for x, y in component:
            img[x, y] = NON_PIECE_VAL

    return img

def defrag_bfs(img, x, y, visited, component):
    pixel_queue = [(x,y)]

    # bfs to find all pixels which haven't already been visited, are real pixels, and are connected
    while len(pixel_queue) > 0:
        x, y = pixel_queue.pop(0)

        if not isInBounds(img, x, y):
            continue

        if visited[x, y] != 0:
            continue

        if areAllNotPiecePixels(img[x, y]):
            continue

        visited[x, y] = 1
        component.add((x, y))
        pixel_queue.append((x, y-1))
        pixel_queue.append((x, y+1))
        pixel_queue.append((x-1, y))
        pixel_queue.append((x+1, y))

def extractPieceBorderImage(img):
    if DEBUG: print('extracting border image')

    w, h, _ = img.shape

    border = -np.ones_like(img)
    bx = -1
    by = -1

    for x in xrange(w):
        for y in xrange(h):
            # make sure it is actually a piece pixel
            if areAllNotPiecePixels(img[x, y]):
                continue

            # if it is on the border of the image, then must be border of piece
            if isOnBorder(img, x, y):
                border[x, y] = 0
                bx = x
                by = y
                continue

            # otherwise make sure it touches a non-piece pixel in a cardinal direction
            for nx, ny in getNeighbors(img, x, y, include_diags = False):
                if areAllNotPiecePixels(img[nx, ny]):
                    border[x, y] = 0
                    bx = x
                    by = y
                    break
    return border, bx, by

def extractPieceBorderPath(img, border_img = None):
    if DEBUG: print('extracting border path')

    if border_img is None:
        border_img, bx, by = extractPieceBorderImage(img)
    else:
        # get an interior point and then shift it over until it is on the edge
        bx, by = getInteriorPoint(img)
        while areAllNotPiecePixels(border_img[bx, by]):
            bx -= 1

    border_path = []

    foundNext = True
    while foundNext:
        border_path.append((bx, by))

        border_img[bx, by] = NON_PIECE_VAL

        foundNext = False

        best_x = -1
        best_y = -1
        best_num_neighbors = 9

        for x, y in getNeighbors(img, bx, by):
            # find all the neighbors of the current pixel which are also on the border
            if areAllPiecePixels(border_img[x, y]):
                # count how many border neighbors this pixel has
                count = 0
                for nx, ny in getNeighbors(img, x, y):
                    if areAllPiecePixels(border_img[nx, ny]):
                        count += 1
                if count < best_num_neighbors:
                    best_num_neighbors = count
                    bx = x
                    by = y
                    foundNext = True

    return border_path

def getPathAngleDeg(border_path, start, end):
    dx = border_path[end][0] - border_path[start][0]
    dy = border_path[end][1] - border_path[start][1]
    return 180 * math.atan2(dy, dx) / math.pi

def extractPieceBorderAngles(border_path):
    angle_list = []
    for i in xrange(len(border_path)):
        j = (i + BORDER_STEP) % len(border_path)
        k = (i - BORDER_STEP + len(border_path)) % len(border_path)

        ang1 = getPathAngleDeg(border_path, i, j)
        ang2 = getPathAngleDeg(border_path, i, k)

        if ang1 < 0:
            ang1 += 360
        if ang2 < 0:
            ang2 += 360

        diff = abs(ang1 - ang2)
        if diff > 180:
            diff = 360 - diff
        angle_list.append(diff)

    return angle_list

def straighten(img):
    if DEBUG: print('straightening piece')
    border_path = extractPieceBorderPath(img)
    angle_list = extractPieceBorderAngles(border_path)

    flat_count = 0
    start_flat_index = -1
    flat_segments = []
    for i in xrange(len(angle_list)):
        if angle_list[i] >= STRAIGHT_SEG_ANGLE_CUTOFF:
            if flat_count == 0:
                start_flat_index = i
            flat_count += 1
        else:
            if flat_count >= STRAIGHT_SEG_LEN_CUTOFF:
                flat_segments.append((start_flat_index, i - 1))
                start_flat_index = -1
            flat_count = 0

    rot_list = []
    for start_ind, end_ind in flat_segments:
        i = (start_ind - BORDER_STEP / 2 + len(border_path)) % len(border_path)
        j = (end_ind + BORDER_STEP / 2) % len(border_path)
        rot_list.append((getPathAngleDeg(border_path, i, j), end_ind - start_ind + BORDER_STEP))

    rot_sum = 0.0
    len_sum = 0
    for rot, seg_len in rot_list:
        num_turns = round((rot - rot_list[0][0]) / 90.0)
        rot_sum += (rot - num_turns * 90.0) * seg_len
        len_sum += seg_len

    return scipy.ndimage.interpolation.rotate(img, -rot_sum / len_sum, order=0, cval=NON_PIECE_VAL)

