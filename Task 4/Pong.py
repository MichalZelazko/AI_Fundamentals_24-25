#!/usr/bin/env python3
# Based on https://python101.readthedocs.io/pl/latest/pygame/pong/#
import pygame
from typing import Type
import skfuzzy as fuzz
import skfuzzy.control as fuzzcontrol
import numpy as np
import matplotlib.pyplot as plt

FPS = 30


class Board:
    def __init__(self, width: int, height: int):
        self.surface = pygame.display.set_mode((width, height), 0, 32)
        pygame.display.set_caption("AIFundamentals - PongGame")

    def draw(self, *args):
        background = (0, 0, 0)
        self.surface.fill(background)
        for drawable in args:
            drawable.draw_on(self.surface)

        pygame.display.update()


class Drawable:
    def __init__(self, x: int, y: int, width: int, height: int, color=(255, 255, 255)):
        self.width = width
        self.height = height
        self.color = color
        self.surface = pygame.Surface(
            [width, height], pygame.SRCALPHA, 32
        ).convert_alpha()
        self.rect = self.surface.get_rect(x=x, y=y)

    def draw_on(self, surface):
        surface.blit(self.surface, self.rect)


class Ball(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        radius: int = 20,
        color=(255, 10, 0),
        speed: int = 3,
    ):
        super(Ball, self).__init__(x, y, radius, radius, color)
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed = speed
        self.y_speed = speed
        self.start_speed = speed
        self.start_x = x
        self.start_y = y
        self.start_color = color
        self.last_collision = 0

    def bounce_y(self):
        self.y_speed *= -1

    def bounce_x(self):
        self.x_speed *= -1

    def bounce_y_power(self):
        self.color = (
            self.color[0],
            self.color[1] + 10 if self.color[1] < 255 else self.color[1],
            self.color[2],
        )
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed *= 1.1
        self.y_speed *= 1.1
        self.bounce_y()

    def reset(self):
        self.rect.x = self.start_x
        self.rect.y = self.start_y
        self.x_speed = self.start_speed
        self.y_speed = self.start_speed
        self.color = self.start_color
        self.bounce_y()

    def move(self, board: Board, *args):
        self.rect.x += round(self.x_speed)
        self.rect.y += round(self.y_speed)

        if self.rect.x < 0 or self.rect.x > (
            board.surface.get_width() - self.rect.width
        ):
            self.bounce_x()

        if self.rect.y < 0 or self.rect.y > (
            board.surface.get_height() - self.rect.height
        ):
            self.reset()

        timestamp = pygame.time.get_ticks()
        if timestamp - self.last_collision < FPS * 4:
            return

        for racket in args:
            if self.rect.colliderect(racket.rect):
                self.last_collision = pygame.time.get_ticks()
                if (self.rect.right < racket.rect.left + racket.rect.width // 4) or (
                    self.rect.left > racket.rect.right - racket.rect.width // 4
                ):
                    self.bounce_y_power()
                else:
                    self.bounce_y()


class Racket(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        width: int = 80,
        height: int = 20,
        color=(255, 255, 255),
        max_speed: int = 10,
    ):
        super(Racket, self).__init__(x, y, width, height, color)
        self.max_speed = max_speed
        self.surface.fill(color)

    def move(self, x: int, board: Board):
        delta = x - self.rect.x
        delta = self.max_speed if delta > self.max_speed else delta
        delta = -self.max_speed if delta < -self.max_speed else delta
        delta = 0 if (self.rect.x + delta) < 0 else delta
        delta = (
            0
            if (self.rect.x + self.width + delta) > board.surface.get_width()
            else delta
        )
        self.rect.x += delta


class Player:
    def __init__(self, racket: Racket, ball: Ball, board: Board) -> None:
        self.ball = ball
        self.racket = racket
        self.board = board

    def move(self, x: int):
        self.racket.move(x, self.board)

    def move_manual(self, x: int):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def act(self, x_diff: int, y_diff: int):
        """
        Do nothing, control is defined in derived classes
        """
        pass


class PongGame:
    def __init__(
        self, width: int, height: int, player1: Type[Player], player2: Type[Player]
    ):
        pygame.init()
        self.board = Board(width, height)
        self.fps_clock = pygame.time.Clock()
        self.ball = Ball(width // 2, height // 2)

        self.opponent_paddle = Racket(x=width // 2, y=0)
        self.oponent = player1(self.opponent_paddle, self.ball, self.board)

        self.player_paddle = Racket(x=width // 2, y=height - 20)
        self.player = player2(self.player_paddle, self.ball, self.board)

    def run(self):
        while not self.handle_events():
            self.ball.move(self.board, self.player_paddle, self.opponent_paddle)
            self.board.draw(
                self.ball,
                self.player_paddle,
                self.opponent_paddle,
            )
            self.oponent.act(
                self.oponent.racket.rect.centerx - self.ball.rect.centerx,
                self.oponent.racket.rect.centery - self.ball.rect.centery,
            )
            self.player.act(
                self.player.racket.rect.centerx - self.ball.rect.centerx,
                self.player.racket.rect.centery - self.ball.rect.centery,
            )
            self.fps_clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                return True
        keys = pygame.key.get_pressed()
        if keys[pygame.constants.K_LEFT]:
            self.player.move_manual(0)
        elif keys[pygame.constants.K_RIGHT]:
            self.player.move_manual(self.board.surface.get_width())
        return False


class NaiveOponent(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(NaiveOponent, self).__init__(racket, ball, board)

    def act(self, x_diff: int, y_diff: int):
        x_cent = self.ball.rect.centerx
        self.move(x_cent)


class HumanPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(HumanPlayer, self).__init__(racket, ball, board)

    def move_manual(self, x: int):
        self.move(x)


# ----------------------------------
# DO NOT MODIFY CODE ABOVE THIS LINE
# ----------------------------------

class FuzzyPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(FuzzyPlayer, self).__init__(racket, ball, board)
        self.setup_mamdami()

    def setup_mamdami(self):
        self.x_dist = fuzzcontrol.Antecedent(np.arange(-800, 801, 2), 'x_dist')
        self.y_dist = fuzzcontrol.Antecedent(np.arange(0, 401, 1), 'y_dist')
        self.velocity = fuzzcontrol.Consequent(np.arange(-20, 21, 1), 'velocity')

        self.x_dist['far_left'] = fuzz.trapmf(self.x_dist.universe, [-800, -800, -600, -200])
        self.x_dist['left'] = fuzz.trapmf(self.x_dist.universe, [-600, -400, -100, 0])
        self.x_dist['center'] = fuzz.trapmf(self.x_dist.universe, [-10, -5, 5, 10])
        self.x_dist['right'] = fuzz.trapmf(self.x_dist.universe, [0, 100, 400, 600])
        self.x_dist['far_right'] = fuzz.trapmf(self.x_dist.universe, [200, 600, 800, 800])

        self.y_dist['far'] = fuzz.trapmf(self.y_dist.universe, [250, 300, 400, 400])
        self.y_dist['medium'] = fuzz.trapmf(self.y_dist.universe, [50, 100, 250, 300])
        self.y_dist['close'] = fuzz.trapmf(self.y_dist.universe, [0, 0, 50, 100])

        self.velocity['fast_left'] = fuzz.trapmf(self.velocity.universe, [-20, -20, -17, -13])
        self.velocity['left'] = fuzz.trapmf(self.velocity.universe, [-17, -13, -8, -5])
        self.velocity['stop'] = fuzz.trapmf(self.velocity.universe, [-3, -1, 1, 3])
        self.velocity['right'] = fuzz.trapmf(self.velocity.universe, [5, 8, 13, 17])
        self.velocity['fast_right'] = fuzz.trapmf(self.velocity.universe, [13, 17, 20, 20])

        rules = [

            fuzzcontrol.Rule(self.x_dist['far_left'] & self.y_dist['close'], self.velocity['fast_right']),
            fuzzcontrol.Rule(self.x_dist['left'] & self.y_dist['close'], self.velocity['right']),
            fuzzcontrol.Rule(self.x_dist['center'] & self.y_dist['close'], self.velocity['stop']),
            fuzzcontrol.Rule(self.x_dist['right'] & self.y_dist['close'], self.velocity['left']),
            fuzzcontrol.Rule(self.x_dist['far_right'] & self.y_dist['close'], self.velocity['fast_left']),

            fuzzcontrol.Rule(self.x_dist['far_left'] & self.y_dist['medium'], self.velocity['fast_right']),
            fuzzcontrol.Rule(self.x_dist['left'] & self.y_dist['medium'], self.velocity['right']),
            fuzzcontrol.Rule(self.x_dist['center'] & self.y_dist['medium'], self.velocity['stop']),
            fuzzcontrol.Rule(self.x_dist['right'] & self.y_dist['medium'], self.velocity['left']),
            fuzzcontrol.Rule(self.x_dist['far_right'] & self.y_dist['medium'], self.velocity['fast_left']),

            fuzzcontrol.Rule(self.x_dist['far_left'] & self.y_dist['far'], self.velocity['fast_right']),
            fuzzcontrol.Rule(self.x_dist['left'] & self.y_dist['far'], self.velocity['right']),
            fuzzcontrol.Rule(self.x_dist['center'] & self.y_dist['far'], self.velocity['stop']),
            fuzzcontrol.Rule(self.x_dist['right'] & self.y_dist['far'], self.velocity['left']),
            fuzzcontrol.Rule(self.x_dist['far_right'] & self.y_dist['far'], self.velocity['fast_left']),
        ]

        self.racket_controller = fuzzcontrol.ControlSystem(rules)
        self.simulation = fuzzcontrol.ControlSystemSimulation(self.racket_controller)

        self.visualize_mfs()

    def act(self, x_diff: int, y_diff: int):
        velocity = self.make_decision(x_diff, y_diff)
        self.move(self.racket.rect.x + int(velocity))

    def make_decision(self, x_diff: int, y_diff: int):
        self.simulation.input['x_dist'] = x_diff
        self.simulation.input['y_dist'] = y_diff
        self.simulation.compute()
        velocity = self.simulation.output.get('velocity', 0)
        return velocity
    
    def visualize_mfs(self):    
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 7))
        
        for name, mf in self.x_dist.terms.items():
            ax1.plot(self.x_dist.universe, mf.mf, label=name)
        ax1.set_title('X Distance Membership Functions')
        ax1.legend()
        ax1.grid(True)
        
        for name, mf in self.y_dist.terms.items():
            ax2.plot(self.y_dist.universe, mf.mf, label=name)
        ax2.set_title('Y Distance Membership Functions')
        ax2.legend()
        ax2.grid(True)
        
        for name, mf in self.velocity.terms.items():
            ax3.plot(self.velocity.universe, mf.mf, label=name)
        ax3.set_title('Velocity Membership Functions')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
    
class FuzzyPlayerTsk(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(FuzzyPlayerTsk, self).__init__(racket, ball, board)
        self.setup_tsk()

    def setup_tsk(self):
        self.x_universe = np.arange(-800, 801, 2)
        self.y_universe = np.arange(0, 401, 1)
        self.velocity_universe = np.arange(-20, 21, 1)

        self.x_mf = {
            "far_left": fuzz.trapmf(self.x_universe, [-800, -800, -600, 0]),
            "left": fuzz.trapmf(self.x_universe, [-600, -500, -300, 0]),
            "center": fuzz.trapmf(self.x_universe, [-30, -10, 10, 30]),
            "right": fuzz.trapmf(self.x_universe, [0, 300, 500, 600]),
            "far_right": fuzz.trapmf(self.x_universe, [0, 600, 800, 800])
        }
    
        self.y_mf = {
            "far": fuzz.trapmf(self.y_universe, [180, 250, 400, 400]),
            "medium": fuzz.trapmf(self.y_universe, [0, 100, 300, 400]),
            "close": fuzz.trapmf(self.y_universe, [0, 0, 80, 200])
        }

        self.velocity_mf = {
            "fast_left": fuzz.trapmf(self.velocity_universe, [-20, -20, -15, -13]),
            "left": fuzz.trapmf(self.velocity_universe, [-15, -13, -7, -2]),
            "stop": fuzz.trapmf(self.velocity_universe, [-3, -1, 1, 3]),
            "right": fuzz.trapmf(self.velocity_universe, [2, 7, 13, 15]),
            "fast_right": fuzz.trapmf(self.velocity_universe, [13, 15, 20, 20])
        }

        self.visualize_mfs()

    def act(self, x_diff: int, y_diff: int):
        velocity = self.make_decision(x_diff, y_diff)
        self.move(self.racket.rect.x + velocity)

    def make_decision(self, x_diff: int, y_diff: int):
        x_vals = {name: fuzz.interp_membership(self.x_universe, mf, x_diff) for name, mf in self.x_mf.items()}
        y_vals = {name: fuzz.interp_membership(self.y_universe, mf, y_diff) for name, mf in self.y_mf.items()}

        activations = {
            "fast_left": max(
                min(x_vals["far_right"], y_vals["far"]), 
                min(x_vals["far_right"], y_vals["medium"]),
                min(x_vals["far_right"], y_vals["close"]), 
                ),
            "left": max(
                min(x_vals["right"], y_vals["far"]), 
                min(x_vals["right"], y_vals["medium"]),
                min(x_vals["right"], y_vals["close"]), 
                ),
            "stop": max(
                min(x_vals["center"], y_vals["far"]),
                min(x_vals["center"], y_vals["medium"]), 
                min(x_vals["center"], y_vals["close"]), 
            ),
            "right": max(
                min(x_vals["left"], y_vals["far"]), 
                min(x_vals["left"], y_vals["medium"]),
                min(x_vals["left"], y_vals["close"]), 
                ),
            "fast_right": max(
                min(x_vals["far_left"], y_vals["far"]), 
                min(x_vals["far_left"], y_vals["medium"]),
                min(x_vals["far_left"], y_vals["close"]), 
                )
        }

        numerator = sum(activations[rule] * fuzz.defuzz(self.velocity_universe, self.velocity_mf[rule], 'centroid') for rule in activations)
        denominator = sum(activations[rule] for rule in activations)
        
        velocity = numerator / denominator if denominator != 0 else 0
        return velocity

    
    def visualize_mfs(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,7))

        for name, mf in self.x_mf.items():
            ax1.plot(self.x_universe, mf, label=name)
        ax1.set_title('X Distance Membership Functions')
        ax1.legend()

        for name, mf in self.y_mf.items():
            ax2.plot(self.y_universe, mf, label=name)
        ax2.set_title('Y Distance Membership Functions')
        ax2.legend()

        for name, mf in self.velocity_mf.items():
            ax3.plot(self.velocity_universe, mf, label=name)
        ax3.set_title('Velocity Membership Functions')
        ax3.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # game = PongGame(800, 400, NaiveOponent, HumanPlayer)
    game = PongGame(800, 400, NaiveOponent, FuzzyPlayer)
    # game = PongGame(800, 400, NaiveOponent, FuzzyPlayerTsk)
    game.run()