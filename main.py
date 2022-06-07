import sys, pygame, pygame.gfxdraw, rnn, geometry, physics, dill
import numpy as np
from pygame.locals import *

pygame.init()

np.random.seed(111)

class food:
    def __init__(self, x, y):
        self.r = 12
        self.energy_full = fps_basis * 60
        self.body = physics.object(x, y, self.r, 0, world_size)
        self.color = np.array([1.0, 0, 0])
        self.own = False
        self.ownership = []
        self.circle = geometry.circle(x, y, self.r)

class blob: #신경망으로도 전달, 생식기능 추가, 오브젝트 처리 범용성, 가계도, 줍기, 버리기
    def __init__(self, x, y, s, d, input_size, hidden_size, output_size, color_inverter_matrix):
        self.energy_full = fps_basis * 60 #hp가 0이 되면 self.hp_full과 같은 hp 보충 능력을 가진 food로 변함
        self.energy = self.energy_full
        self.alive = True
        self.s = s
        self.r = {'male' : 16, 'female' : 24}[self.s]
        self.body = physics.object(x, y, self.r, d, world_size)
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
        self.color_inverter_matrix = color_inverter_matrix
        self.moving_f = 3
        self.rotation_f = 1
        self.view_size = 128
        self.num_ray = 2 #중심 레이 양 옆 레이들의 개수
        self.diff_angle_ray = 10
        self.circle = geometry.circle(x, y, self.r)
        self.message = np.zeros(3)

        if self.s == 'male':
            self.weights = []
            self.weights.append(rnn.get_weight(self.input_size, self.hidden_size))
            self.weights.append(rnn.get_weight(self.hidden_size, self.hidden_size))
            self.weights.append(rnn.get_weight(self.hidden_size, self.output_size))
            self.brain = rnn.nn(self.input_size, self.hidden_size, self.output_size)
            self.brain.reset(*self.weights)
            self.color = rnn.sigmoid(np.dot(self.brain.forward(np.ones(self.input_size)), self.color_inverter_matrix))
            self.brain.reset_memory()

        elif self.s == 'female':
            self.weights = []
            for i in range(0, 2):
                self.weights.append([])
                self.weights[i].append(rnn.get_weight(self.input_size, self.hidden_size))
                self.weights[i].append(rnn.get_weight(self.hidden_size, self.hidden_size))
                self.weights[i].append(rnn.get_weight(self.hidden_size, self.output_size))
            self.selection = []
            selected_weights = []
            for i in range(0, 3):
                tmp = np.random.randint(2)
                self.selection.append(tmp)
                selected_weights.append(self.weights[tmp][i])
            self.brain = rnn.nn(self.input_size, self.hidden_size, self.output_size)
            self.brain.reset(*selected_weights)
            self.color = rnn.sigmoid(np.dot(self.brain.forward(np.ones(self.input_size)), self.color_inverter_matrix))
            self.brain.reset_memory()

    def reset_circle(self):
        self.circle = geometry.circle(self.body.x, self.body.y, self.r)

    def get_ray_line(self):
        rays = []
        for i in range(0, self.num_ray):
            rays.append(geometry.lengthdir_line(self.body.x, self.body.y, self.view_size, self.body.d - (i + 1) * self.diff_angle_ray))
        rays.append(geometry.lengthdir_line(self.body.x, self.body.y, self.view_size, self.body.d))
        for i in range(0, self.num_ray):
            rays.append(geometry.lengthdir_line(self.body.x, self.body.y, self.view_size, self.body.d + (i + 1) * self.diff_angle_ray))
        return rays

class population:
    def __init__(self, initial_population, initial_num_food, input_size, hidden_size, output_size):
        self.initial_population = initial_population
        self.initial_num_food = initial_num_food
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
        self.color_inverter_matrix = np.random.normal(loc = 0.0, scale = 1 / (self.output_size ** 0.5), size = (self.output_size, 3))
        self.interval = 32
        self.blobs = []
        self.foods = []
        structure = (self.input_size, self.hidden_size, self.output_size)
        for i in range(0, self.initial_population):
            self.blobs.append(blob(*population.get_random_point(self.interval), ['male', 'female'][np.random.randint(2)], 0, *structure, self.color_inverter_matrix))
        for i in range(0, self.initial_num_food):
            self.foods.append(food(*population.get_random_point(self.interval)))

    def get_random_point(interval):
        return pygame.Vector2(np.random.randint(interval, world_size[0] - interval + 1), np.random.randint(interval, world_size[1] - interval + 1))

#색상
c_white = (255, 255, 255)
c_black = (0, 0, 0)
c_gray = (128, 128, 128)

#화면
screen_size = (1024, 768)
world_size_m = 6
minimap_scale = 16
world_size = [screen_size[i] * world_size_m for i in range(0, 2)]
display_surf = pygame.display.set_mode(screen_size)

#시간
fps_clock = pygame.time.Clock()
fps = 120
fps_basis = 60

#카메라
camera = pygame.Vector2(0, 0)
camera_moving_delta = 32
system_action = {'camera_up' : False, 'camera_down' : False, 'camera_left' : False, 'camera_right' : False, 'minimap_mouse_handling' : False}
system_switch = {'show_minimap' : True, 'show_all_map' : False}
panel_interval = 8

#기타 설정
moving_friction_size = 0.01
rotation_friction_size = 0.1

pygame.display.set_caption('시뮬레이터')

#blobs
P = population(40, 20, 33, 100, 10)

while True:
    display_surf.fill(c_white)
    #그리기 겸 시뮬레이터 구동
    num_food = len(P.foods)
    num_population = len(P.blobs)
    for i in range(0, num_food): #food만 따로 처리
        if not P.foods[i].own:
            #일반 크기로 화면 보여주기
            if not system_switch['show_all_map']:
                if camera.x - P.foods[i].r <= P.foods[i].body.x <= camera.x + screen_size[0] + P.foods[i].r and camera.y - P.foods[i].r <= P.foods[i].body.y <= camera.y + screen_size[1] + P.foods[i].r:
                    pygame.gfxdraw.aacircle(display_surf, int(P.foods[i].body.x - camera.x), int(P.foods[i].body.y - camera.y), int(P.foods[i].r), P.foods[i].color * 255)
            #전체 화면 보여주기
            else:
                pygame.gfxdraw.aacircle(display_surf, int(P.foods[i].body.x / world_size_m), int(P.foods[i].body.y / world_size_m), int(P.foods[i].r / world_size_m), P.foods[i].color * 255)
    for i in range(0, num_population):
        #감각신호 관련 변수들
        if P.blobs[i].alive:
            tmp_ray_lines_length = P.blobs[i].view_size
            tmp_ray_lines = P.blobs[i].get_ray_line()
            tmp_num_ray_lines = P.blobs[i].num_ray * 2 + 1
            tmp_blob_color = P.blobs[i].color
            tmp_blob_radius = P.blobs[i].r
            tmp_collide_min = [P.blobs[i].view_size for i in range(0, tmp_num_ray_lines)]
            tmp_info = [(np.zeros(3) - tmp_blob_color, 0) for i in range(0, tmp_num_ray_lines)]

        #일반 크기로 화면 보여주기
        if not system_switch['show_all_map']:
            if P.blobs[i].alive and camera.x - P.blobs[i].r <= P.blobs[i].body.x <= camera.x + screen_size[0] + P.blobs[i].r and camera.y - P.blobs[i].r <= P.blobs[i].body.y <= camera.y + screen_size[1] + P.blobs[i].r:
                pygame.gfxdraw.aacircle(display_surf, int(P.blobs[i].body.x - camera.x), int(P.blobs[i].body.y - camera.y), int(P.blobs[i].r), P.blobs[i].color * 255)
        
        #전체 화면 보여주기
        elif P.blobs[i].alive:
            pygame.gfxdraw.aacircle(display_surf, int(P.blobs[i].body.x / world_size_m), int(P.blobs[i].body.y / world_size_m), int(P.blobs[i].r / world_size_m), P.blobs[i].color * 255)
        
        #감각신호 처리, food에 대해서도 처리 넣을 것
        if P.blobs[i].alive:
            for j in range(0, num_population + num_food):
                if j < num_population:
                    if i != j and P.blobs[j].alive and ((P.blobs[i].body.x - P.blobs[j].body.x) ** 2 + (P.blobs[i].body.y - P.blobs[j].body.y) ** 2) ** 0.5 < (P.blobs[i].view_size + P.blobs[j].r):
                        for k in range(0, tmp_num_ray_lines):
                            tmp_collide = geometry.circle_line_collide(P.blobs[j].circle, tmp_ray_lines[k])
                            if tmp_collide != None:
                                if tmp_collide <= tmp_collide_min[k]:
                                    tmp_collide_min[k] = tmp_collide
                                    tmp_info[k] = (P.blobs[j].color - tmp_blob_color, P.blobs[j].r / tmp_blob_radius)
                            else:
                                    tmp_blob_d = P.blobs[i].body.d
                                    tmp_line_end_x, tmp_line_end_y = tmp_ray_lines[k].x2, tmp_ray_lines[k].y2
                                    if 0 <= tmp_blob_d <= 90:
                                        if tmp_line_end_x >= world_size[0] or tmp_line_end_y <= 0:
                                            tmp_collide_wall = min((world_size[0] - P.blobs[i].body.x) / np.abs(np.cos(geometry.dig_to_rad(tmp_blob_d))), P.blobs[i].body.y / np.abs(np.sin(geometry.dig_to_rad(tmp_blob_d))))
                                            tmp_collide_min[k] = tmp_collide_wall
                                    if 90 <= tmp_blob_d <= 180:
                                        if tmp_line_end_x <= 0 or tmp_line_end_y <= 0:
                                            tmp_collide_wall = min(P.blobs[i].body.x / np.abs(np.cos(geometry.dig_to_rad(tmp_blob_d))), P.blobs[i].body.y / np.abs(np.sin(geometry.dig_to_rad(tmp_blob_d))))
                                            tmp_collide_min[k] = tmp_collide_wall
                                    if 180 <= tmp_blob_d <= 270:
                                        if tmp_line_end_x <= 0 or tmp_line_end_y >= world_size[1]:
                                            tmp_collide_wall = min(P.blobs[i].body.x / np.abs(np.cos(geometry.dig_to_rad(tmp_blob_d))), P.blobs[i].body.y / np.abs(np.sin(geometry.dig_to_rad(tmp_blob_d))))
                                            tmp_collide_min[k] = tmp_collide_wall
                                    if 270 <= tmp_blob_d <= 360:
                                        if tmp_line_end_x >= world_size[0] or tmp_line_end_y >= world_size[1]:
                                            tmp_collide_wall = min((world_size[0] - P.blobs[i].body.x) / np.abs(np.cos(geometry.dig_to_rad(tmp_blob_d))), (world_size[1] - P.blobs[i].body.y) / np.abs(np.sin(geometry.dig_to_rad(tmp_blob_d))))
                                            tmp_collide_min[k] = tmp_collide_wall
                else: #food를 처리할 수 있도록 변경
                    food_index = j - num_population
                    if ((P.blobs[i].body.x - P.foods[food_index].body.x) ** 2 + (P.blobs[i].body.y - P.foods[food_index].body.y) ** 2) ** 0.5 < (P.blobs[i].r + P.foods[food_index].r):
                        P.foods[food_index].own = True
                        P.foods[food_index].ownership.append(P.blobs[i])
                    elif not P.foods[food_index].own and ((P.blobs[i].body.x - P.foods[food_index].body.x) ** 2 + (P.blobs[i].body.y - P.foods[food_index].body.y) ** 2) ** 0.5 < (P.blobs[i].view_size + P.foods[food_index].r):
                        for k in range(0, tmp_num_ray_lines):
                            tmp_collide = geometry.circle_line_collide(P.foods[food_index].circle, tmp_ray_lines[k])
                            if tmp_collide != None:
                                if tmp_collide <= tmp_collide_min[k]:
                                    tmp_collide_min[k] = tmp_collide
                                    tmp_info[k] = (P.foods[food_index].color - tmp_blob_color, P.foods[food_index].r / tmp_blob_radius)
                            else:
                                    tmp_blob_d = P.blobs[i].body.d
                                    tmp_line_end_x, tmp_line_end_y = tmp_ray_lines[k].x2, tmp_ray_lines[k].y2
                                    if 0 <= tmp_blob_d <= 90:
                                        if tmp_line_end_x >= world_size[0] or tmp_line_end_y <= 0:
                                            tmp_collide_wall = min((world_size[0] - P.blobs[i].body.x) / np.abs(np.cos(geometry.dig_to_rad(tmp_blob_d))), P.blobs[i].body.y / np.abs(np.sin(geometry.dig_to_rad(tmp_blob_d))))
                                            tmp_collide_min[k] = tmp_collide_wall
                                    if 90 <= tmp_blob_d <= 180:
                                        if tmp_line_end_x <= 0 or tmp_line_end_y <= 0:
                                            tmp_collide_wall = min(P.blobs[i].body.x / np.abs(np.cos(geometry.dig_to_rad(tmp_blob_d))), P.blobs[i].body.y / np.abs(np.sin(geometry.dig_to_rad(tmp_blob_d))))
                                            tmp_collide_min[k] = tmp_collide_wall
                                    if 180 <= tmp_blob_d <= 270:
                                        if tmp_line_end_x <= 0 or tmp_line_end_y >= world_size[1]:
                                            tmp_collide_wall = min(P.blobs[i].body.x / np.abs(np.cos(geometry.dig_to_rad(tmp_blob_d))), P.blobs[i].body.y / np.abs(np.sin(geometry.dig_to_rad(tmp_blob_d))))
                                            tmp_collide_min[k] = tmp_collide_wall
                                    if 270 <= tmp_blob_d <= 360:
                                        if tmp_line_end_x >= world_size[0] or tmp_line_end_y >= world_size[1]:
                                            tmp_collide_wall = min((world_size[0] - P.blobs[i].body.x) / np.abs(np.cos(geometry.dig_to_rad(tmp_blob_d))), (world_size[1] - P.blobs[i].body.y) / np.abs(np.sin(geometry.dig_to_rad(tmp_blob_d))))
                                            tmp_collide_min[k] = tmp_collide_wall

            #신경망에 정보 입력
            input_velocity = P.blobs[i].body.velocity / P.blobs[i].body.velocity_max
            input_rotation_velocity = P.blobs[i].body.rotation_velocity / P.blobs[i].body.rotation_velocity_max
            input_degree = P.blobs[i].body.d / 360
            input_message = P.blobs[i].message.tolist()
            brain_input = [*input_velocity, input_rotation_velocity, input_degree, *input_message, P.blobs[i].energy / P.blobs[i].energy_full]
            for j in range(0, tmp_num_ray_lines):
                tmp_color, tmp_radius = tmp_info[j]
                tmp_color = tmp_color.tolist()
                tmp_distance = tmp_collide_min[j] / tmp_ray_lines_length
                brain_input += tmp_color
                brain_input.append(tmp_radius)
                brain_input.append(tmp_distance)
            brain_output = rnn.sigmoid(P.blobs[i].brain.forward(np.array(brain_input)))

            #출력을 바탕으로 행동
            action_size = 6
            action_prob = brain_output[:action_size]
            send_message_prob = brain_output[action_size]
            message = brain_output[-3:]
            action = np.argmax(action_prob)
            if action == 0: #우회전
                if P.blobs[i].energy > 0:
                    P.blobs[i].body.rotation_force(P.blobs[i].rotation_f)
                    P.blobs[i].energy -= 1
                else:
                    P.blobs[i].alive = False
            if action == 1: #좌회전
                if P.blobs[i].energy > 0:
                    P.blobs[i].body.rotation_force(-P.blobs[i].rotation_f)
                    P.blobs[i].energy -= 1
                else:
                    P.blobs[i].alive = False
            if action == 2: #전진
                if P.blobs[i].energy > 0:
                    P.blobs[i].body.direction_force(P.blobs[i].moving_f)
                    P.blobs[i].energy -= 1
                else:
                    P.blobs[i].alive = False
            if action == 3: #줍기
                pass
            if action == 4: #버리기
                pass
            if send_message_prob > 0.5:
                for j in range(0, num_population):
                    if i != j and ((P.blobs[i].body.x - P.blobs[j].body.x) ** 2 + (P.blobs[i].body.y - P.blobs[j].body.y) ** 2) ** 0.5 < (P.blobs[i].view_size + P.blobs[j].r):
                        P.blobs[j].message = message

            #위치 수정, 기타
            P.blobs[i].reset_circle()
            P.blobs[i].body.frictionize(moving_friction_size)
            P.blobs[i].body.rotation_frictionize(rotation_friction_size)
            P.blobs[i].body.activate()
            P.blobs[i].energy -= 1
            if P.blobs[i].energy <= 0:
                P.blobs[i].alive = False
    
    #미니맵 보여주기
    if system_switch['show_minimap']:
        pygame.draw.rect(display_surf, c_white, (panel_interval, panel_interval, world_size[0] / minimap_scale, world_size[1] / minimap_scale))
        pygame.draw.rect(display_surf, c_gray, (panel_interval + camera.x / minimap_scale, panel_interval + camera.y / minimap_scale, screen_size[0] / minimap_scale, screen_size[1] / minimap_scale), 1)
        pygame.draw.rect(display_surf, c_black, (panel_interval, panel_interval, world_size[0] / minimap_scale, world_size[1] / minimap_scale), 1)
        for i in range(0, num_population):
            if P.blobs[i].alive:
                minimap_blob_size_x = min({'male' : 2, 'female' : 4}[P.blobs[i].s], (world_size[0] - P.blobs[i].body.x) / minimap_scale, P.blobs[i].body.x / minimap_scale)
                minimap_blob_size_y = min({'male' : 2, 'female' : 4}[P.blobs[i].s], (world_size[1] - P.blobs[i].body.y) / minimap_scale, P.blobs[i].body.y / minimap_scale)
                minimap_blob_x = panel_interval + P.blobs[i].body.x / minimap_scale
                minimap_blob_y = panel_interval + P.blobs[i].body.y / minimap_scale
                pygame.draw.rect(display_surf, P.blobs[i].color * 255, (minimap_blob_x, minimap_blob_y, minimap_blob_size_x, minimap_blob_size_y))
        for i in range(0, num_food):
            if not P.foods[i].own:
                minimap_food_size_x = min(5, (world_size[0] - P.foods[i].body.x) / minimap_scale, P.foods[i].body.x / minimap_scale)
                minimap_food_size_y = min(5, (world_size[1] - P.foods[i].body.y) / minimap_scale, P.foods[i].body.y / minimap_scale)
                minimap_food_x = panel_interval + P.foods[i].body.x / minimap_scale
                minimap_food_y = panel_interval + P.foods[i].body.y / minimap_scale
                pygame.draw.rect(display_surf, P.foods[i].color * 255, (minimap_food_x, minimap_food_y, minimap_food_size_x, minimap_food_size_y))

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            k = pygame.key.get_pressed()
            if not system_switch['show_all_map']:
                if k[K_w]:
                    system_action['camera_up'] = True
                if k[K_a]:
                    system_action['camera_left'] = True
                if k[K_s]:
                    system_action['camera_down'] = True
                if k[K_d]:
                    system_action['camera_right'] = True
                if k[K_m]:
                    if system_switch['show_minimap']:
                        system_switch['show_minimap'] = False
                    else:
                        system_switch['show_minimap'] = True
            if k[K_z]:
                if system_switch['show_all_map']:
                    system_switch['show_all_map'] = False
                    system_switch['show_minimap'] = True
                else:
                    system_switch['show_all_map'] = True
                    system_switch['show_minimap'] = False
        if event.type == KEYUP:
            system_action['camera_up'] = False
            system_action['camera_down'] = False
            system_action['camera_left'] = False
            system_action['camera_right'] = False

        #미니맵을 마우스로 조작
        if system_switch['show_minimap']:
            if event.type == MOUSEBUTTONDOWN:
                system_action['minimap_mouse_handling'] = True
                mouse_x, mouse_y = pygame.mouse.get_pos()
                diff_x, diff_y = -screen_size[0] / (2 * minimap_scale) - panel_interval, -screen_size[1] / (2 * minimap_scale) - panel_interval
                if panel_interval + camera.x / minimap_scale <= mouse_x <= panel_interval + (camera.x + screen_size[0]) / minimap_scale and panel_interval + camera.y / minimap_scale <= mouse_y <= panel_interval + (camera.y + screen_size[1]) / minimap_scale:
                    diff_x, diff_y = camera.x / minimap_scale - mouse_x, camera.y / minimap_scale - mouse_y
            if event.type == MOUSEBUTTONUP:
                system_action['minimap_mouse_handling'] = False
            if system_action['minimap_mouse_handling']:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if 0 <= mouse_x + diff_x <= (world_size[0] - screen_size[0]) / minimap_scale:
                    camera.x = (mouse_x + diff_x) * minimap_scale
                if 0 <= mouse_y + diff_y <= (world_size[1] - screen_size[1]) / minimap_scale:
                    camera.y = (mouse_y + diff_y) * minimap_scale
                if 0 > mouse_x + diff_x:
                    camera.x = 0
                if mouse_x + diff_x > (world_size[0] - screen_size[0]) / minimap_scale:
                    camera.x = world_size[0] - screen_size[0]
                if 0 > mouse_y + diff_y:
                    camera.y = 0
                if mouse_y + diff_y > (world_size[1] - screen_size[1]) / minimap_scale:
                    camera.y = world_size[1] - screen_size[1]

    if system_action['camera_up']:
        if camera.y > 0:
            camera.y -= camera_moving_delta
            camera.y = max(camera.y, 0)
    if system_action['camera_down']:
        if camera.y < world_size[1] - screen_size[1]:
            camera.y += camera_moving_delta
            camera.y = min(camera.y, world_size[1] - screen_size[1])
    if system_action['camera_right']:
        if camera.x < world_size[0] - screen_size[0]:
            camera.x += camera_moving_delta
            camera.x = min(camera.x, world_size[0] - screen_size[0])
    if system_action['camera_left']:
        if camera.x > 0:
            camera.x -= camera_moving_delta
            camera.x = max(camera.x, 0)

    pygame.display.update()

    #alive == False인 개체를 리스트에서 flush 하기

    fps_clock.tick(fps)