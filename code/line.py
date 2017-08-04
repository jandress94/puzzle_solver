import math
import sys
import numpy as np

class Line:
    def __init__(self, x, y, angle, weight):
        self.x = x
        self.y = y
        self.angle = angle
        self.weight = weight

    def evalGivenX(self, xVal):
        return (xVal - self.x) * math.tan(math.pi * self.angle / 180) + self.y

    def evalGivenY(self, yVal):
        return (yVal - self.y) / math.tan(math.pi * self.angle / 180) + self.x

    def intersect(self, other_line):
        if self.angle == other_line.angle:
            x = (self.x * self.weight + other_line.x * other_line.weight) / (self.weight + other_line.weight)
            y = (self.y * self.weight + other_line.y * other_line.weight) / (self.weight + other_line.weight)
        else:
            vert_line = None
            horiz_line = None
            non_vert_line = None
            non_horiz_line = None

            if self.angle == 0:
                horiz_line = self
                non_horiz_line = other_line
            elif other_line.angle == 0:
                horiz_line = other_line
                non_horiz_line = self

            if self.angle == 90:
                vert_line = self
                non_vert_line = other_line
            elif other_line.angle == 90:
                vert_line = other_line
                non_vert_line = self

            if horiz_line is not None and vert_line is not None:
                y = horiz_line.y
                x = vert_line.x
            elif horiz_line is not None:
                y = horiz_line.y
                x = non_horiz_line.evalGivenY(y)
            elif vert_line is not None:
                x = vert_line.x
                y = non_vert_line.evalGivenX(x)
            else:
                m1 = math.tan(math.pi * self.angle / 180)
                m2 = math.tan(math.pi * other_line.angle / 180)
                x = (other_line.y - self.y - m2 * other_line.x + m1 * self.x) / (m1 - m2)
                y = self.evalGivenX(x)
        return x, y

    def __str__(self):
        return "Line((%f, %f), %f, %f)" % (self.x, self.y, self.angle, self.weight)

def lineFromCoords(coord1, coord2):
    x1 = coord1[0]
    y1 = coord1[1]
    x2 = coord2[0]
    y2 = coord2[1]

    if x1 == x2 and y1 == y2:
        sys.exit("The two points defining a line must be different: (%d, %d) (%d, %d)" % (x1, y1, x2, y2))
    
    angle = 180 * math.atan2(y2 - y1, x2 - x1) / math.pi
    if angle < -45:
        angle += 360

    return Line(x1, y1, angle, 1.0)

def lineFromLines(line1, line2):
    weight = line1.weight + line2.weight
        
    if line1.angle == line2.angle:
        angle = line1.angle
    else:
        angle = (line1.angle * line1.weight + line2.angle * line2.weight) / weight

    x, y = line1.intersect(line2)

    return Line(x, y, angle, weight)