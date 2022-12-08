import pygame
import numpy as np
import collections
from copy import deepcopy
from Util import *


class Block:
    def __init__(self,
                 rect: pygame.Rect | FakeRect,
                 direction: str):
        self.rect = rect
        self.direction = direction


class History:
    """Helper class for handling the path history of
    the Snake. """

    def __init__(self, length: int):
        self.length = length
        self.history = collections.deque([], maxlen=self.length)

    def add(self, state: tuple[int | float, int | float, str]) -> None:
        self.history.append(state)

    def get(self, index: int) -> tuple[int | float, int | float, str]:
        return self.history.__getitem__(index)

    def set_length(self, new_length: int) -> None:
        assert new_length >= self.history.__len__(), f'New length should be >= current.'
        self.history = collections.deque(list(deepcopy(self.history)), maxlen=new_length)


class Apple:
    def __init__(self,
                 rect: pygame.Rect | FakeRect,
                 screen_size: int,
                 seed: int):
        assert type(screen_size) is int, f'Screen size should be given as int, but is: {screen_size}'
        assert screen_size > 2 * rect.size[0], f'Screen size should be larger compared to rect size.'

        # For RNG reproducibility
        np.random.seed(seed)

        self._rect = rect
        self._block_size = self._rect.size[0]
        self._screen_size = screen_size

        self.apple_block = None

    def _add_apple_block(self, grid: np.ndarray) -> None:
        """Setting apple block at random place in grid
        that is not occupied by snake."""

        assert len(grid.shape) == 2, 'grid should be 2D array.'
        assert grid.shape[0] == grid.shape[1], 'grid should be symmetric.'

        mask = grid == 0
        indices_grids = np.indices(grid.shape) * self._block_size  # 2 x grid height x grid width
        available_x, available_y = list(indices_grids[0][mask]), list(indices_grids[1][mask])
        indices = np.array([available_x, available_y])  # 2 x (grid height - snake length)
        coordinates = indices[:, np.random.randint(low=0, high=len(available_y))]
        rect = deepcopy(self._rect)
        rect.left, rect.top = coordinates[0], coordinates[1]
        apple_block = Block(rect=rect, direction="Irrelevant")
        self.apple_block = apple_block

    def initialize(self, grid: np.ndarray) -> None:
        """ Public wrapper function for generating
        apple block. """
        self._add_apple_block(grid=grid)

    def get_apple(self):
        """ Getter function for apple rect. """
        return self.apple_block.rect


class Snake:
    """ In this implementation it is assumed that
        'positive x' is to the right and that
        'positive y' is downwards.

         General procedure:
          1) Generate snake head
          2) save history
          for i in range...
                - do stuff (spawn in body etc.)
                - move
                - save history
         """

    def __init__(self,
                 rect: pygame.Rect | FakeRect,
                 screen_size: int,
                 seed: int) -> None:

        assert type(screen_size) is int, f'Screen size should be given as int, but is: {screen_size}'
        assert screen_size > 2 * rect.size[0], f'Screen size should be larger compared to rect size.'

        # For RNG reproducibility
        np.random.seed(seed)

        self._rect = rect
        self._block_size = self._rect.size[0]
        self._screen_size = screen_size
        self._snake_blocks = []
        self._action_space = ["up", "down", "left", "right"]

        self.snake_length = 0
        self.dead = False
        self.history = History(length=self.snake_length)

    def _update_snake_length(self) -> None:
        """ Updating length of snake and history """
        self.snake_length = len(self._snake_blocks)
        self.history.set_length(self.snake_length)

    def _add_head_block(self, direction: str = "Unknown") -> None:
        """ Assuming symmetric block size and screen size"""
        assert self.snake_length == 0, f'Snake should be 0 length for spawning head, but is {self.snake_length}.'
        assert self._screen_size % self._block_size == 0, f'Screen size should be integer multiple of block size.'
        if direction == "Unknown":
            direction = np.random.choice(self._action_space)
        rect = deepcopy(self._rect)
        grid_size = self._screen_size // self._block_size
        rect.left, rect.top = np.random.randint(low=0, high=grid_size, size=2) * self._block_size
        head_block = Block(rect=rect, direction=direction)
        self._snake_blocks.append(head_block)
        self._update_snake_length()

    def _add_body_block(self) -> None:
        """ Assuming symmetric block size and screen size"""
        rect = deepcopy(self._rect)
        _map = {"up": (0, self._block_size), "down": (0, -self._block_size),
                "left": (self._block_size, 0), "right": (-self._block_size, 0)}
        rect.left, rect.bottom, direction = self.history.get(0)
        rect.left += _map[direction][0]
        rect.bottom += _map[direction][1]
        body_block = Block(rect=rect, direction=direction)
        self._snake_blocks.append(body_block)
        self._update_snake_length()

    def _move(self, velocity: int | float) -> None:
        """ Assuming v_x == v_y """
        assert self.snake_length > 0, f'The snake should be at least 1 long.'
        # Moving head
        _map = {"up": (0, -velocity), "down": (0, velocity),
                "left": (-velocity, 0), "right": (velocity, 0)}
        _dir = self._snake_blocks[0].direction
        self._snake_blocks[0].rect = self._snake_blocks[0].rect.move(_map[_dir][0], _map[_dir][1])

        # Moving body
        for block in range(1, self.snake_length):
            _x, _y, _dir = self.history.get(-(self.snake_length - block))
            self._snake_blocks[block].rect.left = _x
            self._snake_blocks[block].rect.bottom = _y
            self._snake_blocks[block].direction = _dir

    def set_direction(self, direction: str) -> None:
        """ Setting the direction of the snake
        by changing the direction of the head block."""
        self._snake_blocks[0].direction = direction

    def get_direction(self) -> str:
        """ Getter function for current snake direction."""
        return self._snake_blocks[0].direction

    def get_head(self) -> pygame.Rect | FakeRect:
        """ Getter function for snake head rect."""
        return self._snake_blocks[0].rect

    def get_body(self, index: int) -> pygame.Rect | FakeRect:
        """ Getter function for snake body rect."""
        return self._snake_blocks[index].rect

    def initialize(self, direction: str = "Unknown") -> None:
        """ initializes snake by spawning in snake head at
            random place, optionally with random direction."""
        if direction == "Unknown":
            direction = np.random.choice(self._action_space)
        self._add_head_block(direction=direction)
        self._save_history()

    def _save_history(self) -> None:
        """ Saving history of snake head path. """
        state = (self._snake_blocks[0].rect.left,
                 self._snake_blocks[0].rect.bottom,
                 self._snake_blocks[0].direction)
        self.history.add(state=state)

    def get_grid(self):
        grid_size = self._screen_size // self._block_size
        state = np.zeros(shape=(grid_size, grid_size), dtype=int)
        head_x = self._snake_blocks[0].rect.left // self._block_size - 1
        head_y = self._snake_blocks[0].rect.top // self._block_size - 1
        state[head_y][head_x] = 2
        for block in range(1, self.snake_length):
            body_x = self._snake_blocks[block].rect.left // self._block_size - 1
            body_y = self._snake_blocks[block].rect.top // self._block_size - 1
            state[body_y][body_x] = 1
        return state

    def view_state(self) -> None:
        state = self.get_grid()
        print(state)

    def _colliding_with_wall(self) -> bool:
        """ Function for checking whether snake has collided with
        either of the walls, according to current position,
        screen size and block size. """
        collision_flag = False
        if self._snake_blocks[0].rect.top < 0 or self._snake_blocks[0].rect.bottom > self._screen_size:
            collision_flag = True
        if self._snake_blocks[0].rect.left < 0 or self._snake_blocks[0].rect.right > self._screen_size:
            collision_flag = True
        return collision_flag

    def _colliding_with_self(self) -> bool:
        """ Function for checking whether snake has collided with
        itself, according to current position and block size. """
        collision_flag = False
        for block in range(1, self.snake_length):
            if self._snake_blocks[block].rect.left == self._snake_blocks[0].rect.left:
                if self._snake_blocks[block].rect.bottom == self._snake_blocks[0].rect.bottom:
                    collision_flag = True
        return collision_flag

    def found_apple(self, apple: Apple) -> bool:
        """ A function for checking whether snake hos
        collided with apple. """
        if self._snake_blocks[0].rect.left == apple.apple_block.rect.left:
            if self._snake_blocks[0].rect.bottom == apple.apple_block.rect.bottom:
                # If snake found apple - add to snake tail and the spawn new apple.
                self._add_body_block()
                apple.initialize(grid=self.get_grid())
                return True

    def update(self, velocity: int | float) -> None:
        """Wrapper function that calls move, checks for collisions
           and saves history."""
        self._move(velocity=velocity)
        if self._colliding_with_self() or self._colliding_with_wall():
            self.dead = True
            return None
        self._save_history()
