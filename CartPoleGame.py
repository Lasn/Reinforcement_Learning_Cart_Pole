import math
import pygame
from pygame.locals import QUIT, KEYDOWN, K_LEFT, K_RIGHT
import numpy as np
import time
import random


class CartPoleSimulation:
    def __init__(self):
        # Constants
        self.GRAVITY = 9.8  # m/s2
        self.POLE_LENGTH = 1.0  # m
        self.POLE_MASS = 0.1  # kg
        self.CART_MASS = 1.0  # kg
        self.TOTAL_MASS = self.POLE_MASS + self.CART_MASS
        self.HALF_POLE_LENGTH = self.POLE_LENGTH / 2.0
        self.POLE_MOMENT = self.POLE_MASS * self.HALF_POLE_LENGTH
        self.CART_FORCE = 2.0

        # Pygame constants
        self.WINDOW_SIZE = (800, 480)
        self.FPS = 60
        self.SCALE = 200
        self.CART_WIDTH = 40  # px

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (31, 119, 180)
        self.ORANGE = (255, 127, 14)

        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("CartPole")
        pygame.display.set_icon(pygame.Surface((1, 1)))

        self.screen = pygame.display.set_mode(self.WINDOW_SIZE, flags=pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        self.score = 0

        # Simulation variables
        self.reset()

    def _get_score(self):
        return time.time() - self.score

    def _draw_world(self):
        self.screen.fill(self.WHITE)

        cart_x = self.WINDOW_SIZE[0] / 2.0 + self.cart * self.SCALE
        cart_y = self.WINDOW_SIZE[1] / 2.0
        pygame.draw.rect(
            self.screen,
            self.BLUE,
            pygame.Rect(cart_x - self.CART_WIDTH // 2, cart_y, self.CART_WIDTH, 20),
        )

        pole_x = cart_x + self.HALF_POLE_LENGTH * self.SCALE * math.sin(self.pole_angle)
        pole_y = cart_y - self.HALF_POLE_LENGTH * self.SCALE * math.cos(self.pole_angle)
        pygame.draw.line(
            self.screen, self.ORANGE, (cart_x, cart_y), (pole_x, pole_y), 5
        )

        pygame.display.flip()

    def reset(self):
        self.cart = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = 0.0  # random.uniform(-0.05, 0.05)
        self.pole_angular_velocity = 0.0
        self.force = 0.0
        self.score = time.time()
        return self._get_state()

    def _get_state(self):
        return np.array(
            [
                self.cart,
                self.cart_velocity,
                self.pole_angle,
                self.pole_angular_velocity,
                self.force,
            ]
        )

    def step(self, action):
        if action == 0:
            self.force -= self.CART_FORCE
        elif action == 1:
            self.force += self.CART_FORCE

        sin_theta = math.sin(self.pole_angle)
        cos_theta = math.cos(self.pole_angle)

        pole_acceleration = (
            self.GRAVITY * sin_theta
            - cos_theta
            * (
                self.force
                + self.POLE_MASS
                * self.HALF_POLE_LENGTH
                * self.pole_angular_velocity**2
                * sin_theta
            )
            / self.TOTAL_MASS
        ) / (
            self.HALF_POLE_LENGTH
            * (4.0 / 3.0 - self.POLE_MASS * cos_theta**2 / self.TOTAL_MASS)
        )
        cart_acceleration = (
            self.force
            + self.POLE_MOMENT
            * (
                self.pole_angular_velocity**2 * sin_theta
                - pole_acceleration * cos_theta
            )
        ) / self.TOTAL_MASS

        self.cart_velocity += cart_acceleration / self.FPS
        self.cart += self.cart_velocity / self.FPS

        self.pole_angular_velocity += pole_acceleration / self.FPS
        self.pole_angle += self.pole_angular_velocity / self.FPS

        # Normalize pole angle
        self.pole_angle = self.pole_angle % (2.0 * math.pi)
        if self.pole_angle > math.pi:
            self.pole_angle -= 2.0 * math.pi
        elif self.pole_angle < -math.pi:
            self.pole_angle += 2.0 * math.pi

        # Pole angle limits
        if self.pole_angle > math.pi / 2.0 or self.pole_angle < -math.pi / 2.0:
            return self._get_state(), -50, True, self._get_score()

        # Collision detection
        if (self.cart * self.SCALE - self.CART_WIDTH / 2) < -self.WINDOW_SIZE[0] / 2.0:
            return self._get_state(), -50, True, self._get_score()

        elif (self.cart * self.SCALE + self.CART_WIDTH / 2) > self.WINDOW_SIZE[0] / 2.0:
            return self._get_state(), -50, True, self._get_score()

        self._draw_world()

        # Reward
        reward = 20.0 - abs(self.pole_angular_velocity)

        return self._get_state(), reward, False, self._get_score()

    def run_simulation(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
                elif event.type == KEYDOWN:
                    if event.key == K_LEFT:
                        self.force -= self.CART_FORCE
                    elif event.key == K_RIGHT:
                        self.force += self.CART_FORCE

            sin_theta = math.sin(self.pole_angle)
            cos_theta = math.cos(self.pole_angle)

            pole_acceleration = (
                self.GRAVITY * sin_theta
                - cos_theta
                * (
                    self.force
                    + self.POLE_MASS
                    * self.HALF_POLE_LENGTH
                    * self.pole_angular_velocity**2
                    * sin_theta
                )
                / self.TOTAL_MASS
            ) / (
                self.HALF_POLE_LENGTH
                * (4.0 / 3.0 - self.POLE_MASS * cos_theta**2 / self.TOTAL_MASS)
            )
            cart_acceleration = (
                self.force
                + self.POLE_MOMENT
                * (
                    self.pole_angular_velocity**2 * sin_theta
                    - pole_acceleration * cos_theta
                )
            ) / self.TOTAL_MASS

            self.cart_velocity += cart_acceleration / self.FPS
            self.cart += self.cart_velocity / self.FPS

            self.pole_angular_velocity += pole_acceleration / self.FPS
            self.pole_angle += self.pole_angular_velocity / self.FPS

            # Normalize pole angle
            self.pole_angle = self.pole_angle % (2.0 * math.pi)
            if self.pole_angle > math.pi:
                self.pole_angle -= 2.0 * math.pi
            elif self.pole_angle < -math.pi:
                self.pole_angle += 2.0 * math.pi

            # Pole angle limits
            if self.pole_angle > math.pi / 2.0 or self.pole_angle < -math.pi / 2.0:
                return

            # Collision detection
            if (self.cart * self.SCALE - self.CART_WIDTH / 2) < -self.WINDOW_SIZE[
                0
            ] / 2.0:
                return
                self.cart = (
                    -self.WINDOW_SIZE[0] / 2 + self.CART_WIDTH / 2
                ) / self.SCALE
                self.cart_velocity = -self.cart_velocity
            elif (self.cart * self.SCALE + self.CART_WIDTH / 2) > self.WINDOW_SIZE[
                0
            ] / 2.0:
                return
                self.cart = (self.WINDOW_SIZE[0] / 2 - self.CART_WIDTH / 2) / self.SCALE
                self.cart_velocity = -self.cart_velocity

            self._draw_world()
            print(self._get_state())
            self.clock.tick(self.FPS)


if __name__ == "__main__":
    sim = CartPoleSimulation()
    sim.run_simulation()
    # running = True
    # while running:
    #     action = random.randint(0, 1)
    #     end = sim.step(action)
    #     if end == False:
    #         running = False
