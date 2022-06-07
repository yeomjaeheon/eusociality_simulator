"""
기능 : 
라디안 - 도 변환
각도 - 길이로부터 벡터 반환
원-선분 충돌 여부 및 충돌 지점까지의 거리 탐지
"""

import numpy as np

def dig_to_rad(dig):
    return dig / 180 * np.pi

def rad_to_dig(rad):
    return rad / np.pi * 180

def lengthdir_line(x, y, length, dig):
    rad = dig_to_rad(dig)
    return line(x, y, x + np.cos(rad) * length, y + np.sin(rad) * length)

def circle_line_collide(circle, line): #충돌하지 않는 경우는 None 반환
    v_AC = vector(circle.x - line.x1, circle.y - line.y1)
    v_BC = vector(circle.x - line.x2, circle.y - line.y2)
    v_AB = vector(line.x2 - line.x1, line.y2 - line.y1)
    v_BA = mult(v_AB, -1)
    cos_theta_1 = dot_product(v_AC, v_AB) / (v_AC.size * v_AB.size)
    cos_theta_2 = dot_product(v_BC, v_BA) / (v_BC.size * v_BA.size)
    h = ((1 - cos_theta_1 ** 2) ** 0.5) * v_AC.size
    if v_AC.size <= circle.r:
        return 1e-3
    elif 0 <= cos_theta_1 <= 1 and 0 <= cos_theta_2 <= 1:
        if h == circle.r:
            return cos_theta_1 * v_AC.size
        elif h < circle.r:
            sin_theta_3 = h / circle.r
            return cos_theta_1 * v_AC.size - ((1 - sin_theta_3 ** 2) ** 0.5) * circle.r
    else:
        if v_BC.size <= circle.r:
            sin_theta_3 = h / circle.r
            return cos_theta_1 * v_AC.size - ((1 - sin_theta_3 ** 2) ** 0.5) * circle.r
    return None

def get_intersection_point(line, collide):
    v_AB = vector(line.x2 - line.x1, line.y2 - line.y1)
    norm_v_AB = mult(v_AB, 1 / v_AB.size * collide)
    return (line.x1 + norm_v_AB.x, line.y1 + norm_v_AB.y)

def dot_product(v1, v2):
    return ((v1.x * v2.x) + (v1.y * v2.y))

def add(v1, v2):
    return vector(v1.x + v2.x, v1.y + v2.y)

def mult(v1, n):
    return vector(v1.x * n, v1.y * n)

class vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = (x ** 2 + y ** 2) ** 0.5

class circle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

class line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2