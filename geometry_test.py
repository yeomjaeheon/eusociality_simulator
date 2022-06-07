import pygame, pygame.gfxdraw, sys, random, geometry
from pygame.locals import *

pygame.init()
window_size = (800, 600)
display_surf = pygame.display.set_mode(window_size)
pygame.display.set_caption('geometry_test')

A = (100, 100)
B = (300, 300)
C = (200, 200)
r = 200

mouse_pressed = False

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if pygame.mouse.get_pressed()[0]:
            mouse_pressed = True
        elif mouse_pressed:
            A = (random.randint(0, window_size[0]), random.randint(0, window_size[1]))
            B = (random.randint(0, window_size[0]), random.randint(0, window_size[1]))
            C = (random.randint(0, window_size[0]), random.randint(0, window_size[1]))
            mouse_pressed = False

        display_surf.fill((255, 255, 255))
        circle = geometry.circle(*C, r)
        line = geometry.line(*A, *B)
        collide = geometry.circle_line_collide(circle, line)
        if  collide != None: #평행하는 경우에 오류 안 발생하는지 확인해볼 것
            target_coord = geometry.get_intersection_point(line, collide)
            pygame.gfxdraw.aacircle(display_surf, *list(map(int, target_coord)), 5, (0, 255, 0))

        pygame.draw.line(display_surf, (255, 0, 0), A, B)
        pygame.gfxdraw.aacircle(display_surf, *list(map(int, A)), 10, (255, 0, 0))
        pygame.gfxdraw.aacircle(display_surf, *list(map(int, C)), r, (0, 0, 0))

        pygame.display.update()