import numpy as np
import random
import math
import scipy.ndimage
import sys

from constants import *
from line import *

def mark(img, x, y, color = np.array([0, 0, 0]), includeNeighbors = True):
    if isInBounds(img, x, y):
        img[x, y] = color
    if includeNeighbors:
        for tx, ty in getNeighbors(img, x, y):
            img[tx, ty] = color

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

def getQuadrant(img, x, y):
    if x > img.shape[0] / 2:
        if y > img.shape[1] / 2: return 3
        else: return 2
    else:
        if y > img.shape[1] / 2: return 0
        else: return 1

def drawLine(img, line, color = np.array([0, 0, 0])):
    mostly_vert = abs(math.tan(math.pi * line.angle / 180)) >= 1
    if mostly_vert:
        upper_bound_input = img.shape[1]
        upper_bound_output = img.shape[0]
        eval_fn = line.evalGivenY
    else:
        upper_bound_input = img.shape[0]
        upper_bound_output = img.shape[1]
        eval_fn = line.evalGivenX

    for i in xrange(upper_bound_input):
        out = round(eval_fn(i))
        if out >= 0 and out < upper_bound_output:
            if mostly_vert:
                img[out, i] = color
            else:
                img[i, out] = color

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

    border = NON_PIECE_VAL * np.ones_like(img)
    border_pix = set()

    for x in xrange(w):
        for y in xrange(h):
            # make sure it is actually a piece pixel
            if areAllNotPiecePixels(img[x, y]):
                continue

            # if it is on the border of the image, then must be border of piece
            if isOnBorder(img, x, y):
                border[x, y] = 0
                border_pix.add((x,y))
                continue

            # otherwise make sure it touches a non-piece pixel in a cardinal direction
            for nx, ny in getNeighbors(img, x, y, include_diags = False):
                if areAllNotPiecePixels(img[nx, ny]):
                    border[x, y] = 0
                    border_pix.add((x,y))
                    break

    ix, iy = getInteriorPoint(img)
    pixel_queue = [(ix, iy)]

    visited = np.zeros((img.shape[0], img.shape[1]))
    bx = -1
    by = -1
    while len(pixel_queue) > 0:
        x, y = pixel_queue.pop(0)

        if not isInBounds(img, x, y):
            continue

        if visited[x, y] != 0:
            continue
        visited[x, y] = 1

        # if this is a border pixel and it is the first time we are seeing it, remove it so that it isn't cleared
        if (x, y) in border_pix:
            border_pix.remove((x, y))
            bx = x
            by = y
        # if this is an interior non-piece pixel, add its neighbors
        elif areAllNotPiecePixels(border[x, y]):
            pixel_queue.append((x, y-1))
            pixel_queue.append((x, y+1))
            pixel_queue.append((x-1, y))
            pixel_queue.append((x+1, y))

    for x, y in border_pix:
        border[x, y] = NON_PIECE_VAL

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

    if abs(border_path[0][0] - border_path[-1][0]) + abs(border_path[0][1] - border_path[-1][1]) > 2:
        border_img, _, _ = extractPieceBorderImage(img)
        x, y = border_path[-1]

        result = ""
        for tx in xrange(max(0, x - 5), min(x + 5, img.shape[0])):
            for ty in xrange(max(0, y - 5), min(y + 5, img.shape[1])):
                if tx == x and ty == y:
                    result += 'X'
                elif areAllPiecePixels(border_img[tx, ty]):
                    result += '+'
                else:
                    result += 'O'
            result += '\n'
        print(result)
        sys.exit("The border doesn't make a loop: %s %s" % (str(border_path[0]), str(border_path[-1])))

    startQuad = getQuadrant(img, border_path[0][0], border_path[0][1])
    index = 1
    while getQuadrant(img, border_path[index][0], border_path[index][1]) == startQuad:
        index += 1
    if getQuadrant(img, border_path[index][0], border_path[index][1]) != (startQuad + 1) % 4:
        border_path.reverse()

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

def getStraightSegments(angle_list):
    flat_count = 0
    start_flat_index = -1
    straight_segments = []
    for i in xrange(len(angle_list)):
        if angle_list[i] >= STRAIGHT_SEG_ANGLE_CUTOFF:
            if flat_count == 0:
                start_flat_index = i
            flat_count += 1
        else:
            if flat_count >= STRAIGHT_SEG_LEN_CUTOFF:
                straight_segments.append((start_flat_index, i - 1))
                start_flat_index = -1
            flat_count = 0
    return straight_segments

def straighten(img):
    if DEBUG: print('straightening piece')
    border_path = extractPieceBorderPath(img)
    angle_list = extractPieceBorderAngles(border_path)
    straight_segments = getStraightSegments(angle_list)

    rot_list = []
    for start_ind, end_ind in straight_segments:
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

def getPieceCorners(img):
    if DEBUG: print("computing piece corners")

    expanded_img = NON_PIECE_VAL * np.ones((img.shape[0] + 10, img.shape[1] + 10, 3))
    expanded_img[5:5+img.shape[0], 5:5+img.shape[1]] = img
    img = expanded_img

    border_path = extractPieceBorderPath(img)
    angle_list = extractPieceBorderAngles(border_path)
    straight_segments = getStraightSegments(angle_list)

    edges = [[] for _ in xrange(4)]

    for start_ind, end_ind in straight_segments:
        i = (start_ind - BORDER_STEP / 2 + len(border_path)) % len(border_path)
        j = (end_ind + BORDER_STEP / 2) % len(border_path)

        vert = abs(border_path[i][0] - border_path[j][0]) < abs(border_path[i][1] - border_path[j][1])

        index = 0 if vert else 1
        if vert:
            x = (border_path[i][0] + border_path[j][0]) / 2.0
            is_upper_left =  x < img.shape[0] / 2.0
            is_upper_right_based_on_dirr = border_path[i][1] - border_path[j][1] < 0
        else:
            y = (border_path[i][1] + border_path[j][1]) / 2.0
            is_upper_left =  y < img.shape[1] / 2.0
            is_upper_right_based_on_dirr = border_path[i][0] - border_path[j][0] < 0

        index += 0 if is_upper_left else 2

        if is_upper_right_based_on_dirr == (index == 1 or index == 2):
            edges[index].append((i, j))

        mark(img, border_path[i][0], border_path[i][1])
        mark(img, border_path[j][0], border_path[j][1])
    mark(img, border_path[0][0], border_path[0][1], color = np.array([0, 255, 0]))


    # edges = [[] for _ in xrange(4)]
    # curr_edge_ind = -1
    # curr_vert = True

    # for start_ind, end_ind in straight_segments:
    #     i = (start_ind - BORDER_STEP / 2 + len(border_path)) % len(border_path)
    #     j = (end_ind + BORDER_STEP / 2) % len(border_path)

    #     vert = abs(border_path[i][0] - border_path[j][0]) < abs(border_path[i][1] - border_path[j][1])

    #     if curr_edge_ind < 0 or curr_vert != vert:
    #         curr_edge_ind += 1
    #         curr_vert = vert

    #     edges[curr_edge_ind % len(edges)].append((i,j))
    #     mark(img, border_path[i][0], border_path[i][1])
    #     mark(img, border_path[j][0], border_path[j][1])
    # mark(img, border_path[0][0], border_path[0][1], color = np.array([0, 255, 0]))

    # print(edges)

    if np.prod([len(x) for x in edges]) == 0:
        sys.exit("At least one of the edge lists was empty: %s" % (str(edges)))

    lines = [[lineFromCoords(border_path[i], border_path[j]) for i, j in edge_list] for edge_list in edges]

    for i in xrange(len(lines)):
        while len(lines[i]) > 1:
            line1 = lines[i].pop(0)
            line2 = lines[i].pop(0)
            lines[i].append(lineFromLines(line1, line2))

    lines = [x[0] for x in lines]

    corners = []
    for i in xrange(len(lines)):
        line1 = lines[i]
        line2 = lines[(i + 1) % len(lines)]

        x, y = line1.intersect(line2)
        x = round(x)
        y = round(y)
        corners.append((x,y))
        # print("(%d, %d)" % (x, y))
        # print(str(line1))

        # mark(img, x, y)
        drawLine(img, line1)

    closest_corner_border_indices = {i: None for i in xrange(len(corners))}


    return img
