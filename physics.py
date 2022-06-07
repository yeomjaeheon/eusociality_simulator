import pygame
import numpy as np

class object:
    def __init__(self, x, y, r, d, world_size):
        self.x, self.y, self.r, self.d = x, y, r, d
        self.world_size = world_size
        self.velocity = pygame.Vector2(0, 0)
        self.rotation_velocity = 0
        self.velocity_max = 5
        self.rotation_velocity_max = 10

    def force(self, f):
        self.velocity += f

    def direction_force(self, f_size):
        f = pygame.Vector2(np.cos(self.d / 180 * np.pi), np.sin(self.d / 180 * np.pi)) * f_size
        self.force(f)

    def rotation_force(self, f):
        self.rotation_velocity += f

    def frictionize(self, friction_size):
        velocity_size = self.velocity.magnitude()
        if velocity_size <= friction_size:
            self.velocity = pygame.Vector2(0, 0)
        else:
            self.velocity = self.velocity.normalize() * (velocity_size - friction_size)

    def rotation_frictionize(self, friction_size):
        abs_rotation_velocity = np.abs(self.rotation_velocity)
        if abs_rotation_velocity > friction_size:
            self.rotation_velocity = self.rotation_velocity / abs_rotation_velocity * (abs_rotation_velocity - friction_size)
        else:
            self.rotation_velocity = 0

    def activate(self):
        if self.velocity.magnitude() > self.velocity_max:
            self.velocity.normalize_ip()
            self.velocity *= self.velocity_max
        abs_rotation_velocity = np.abs(self.rotation_velocity)
        if abs_rotation_velocity > self.rotation_velocity_max:
            self.rotation_velocity = self.rotation_velocity / abs_rotation_velocity * self.rotation_velocity_max
        if self.x < self.r:
            self.force(pygame.Vector2(self.velocity_max, 0))
        if self.x > self.world_size[0] - self.r:
            self.force(pygame.Vector2(-self.velocity_max, 0))
        if self.y < self.r:
            self.force(pygame.Vector2(0, self.velocity_max))
        if self.y > self.world_size[1] - self.r:
            self.force(pygame.Vector2(0, -self.velocity_max))
        self.x += self.velocity.x
        self.y += self.velocity.y
        self.d += self.rotation_velocity
        self.d %= 360
        