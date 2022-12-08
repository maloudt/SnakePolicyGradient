import numpy as np
import torch
import pygame


class FakeRect:
    """ Imitation type class implementation of Pygame Reacts.
        Needs 'width' and 'height' for initialization, but auto-adjusts,
        remaining params when one is adjusted and implements
        'property' type setter/getter attributes."""

    def __init__(self, height, width,
                 centerx: int = 0,
                 centery: int = 0,
                 left: int = 0,
                 right: int = 0,
                 top: int = 0,
                 bottom: int = 0):
        self._height = height
        self._width = width
        self._centerx = centerx
        self._centery = centery
        self._left = left
        self._right = right
        self._top = top
        self._bottom = bottom
        self.size = (self._height, self._width)

    def _get_right(self):
        return self._right

    def _set_right(self, value):
        self._right = value
        self._left = self._right - self._width
        self._centerx = self._right - self._width // 2

    def _del_right(self):
        del self._right

    right = property(
        fget=_get_right,
        fset=_set_right,
        fdel=_del_right,
        doc="The right property."
    )

    def _get_left(self):
        return self._left

    def _set_left(self, value):
        self._left = value
        self._right = self._left + self._width
        self._centerx = self._left + self._width // 2

    def _del_left(self):
        del self._left

    left = property(
        fget=_get_left,
        fset=_set_left,
        fdel=_del_left,
        doc="The left property."
    )

    def _get_centerx(self):
        return self._centerx

    def _set_centerx(self, value):
        self._centerx = value
        self._left = self._centerx - self._width // 2
        self._right = self._left + self._width

    def _del_centerx(self):
        del self._centerx

    centerx = property(
        fget=_get_centerx,
        fset=_set_centerx,
        fdel=_del_centerx,
        doc="The centerx property."
    )

    def _get_centery(self):
        return self._centery

    def _set_centery(self, value):
        self._centery = value
        self._top = self._centery - self._height // 2
        self._bottom = self._top + self._height

    def _del_centery(self):
        del self._centery

    centery = property(
        fget=_get_centery,
        fset=_set_centery,
        fdel=_del_centery,
        doc="The centery property."
    )

    def _get_top(self):
        return self._top

    def _set_top(self, value):
        self._top = value
        self._centery = self._top + self._height // 2
        self._bottom = self._top + self._height

    def _del_top(self):
        del self._top

    top = property(
        fget=_get_top,
        fset=_set_top,
        fdel=_del_top,
        doc="The top property."
    )

    def _get_bottom(self):
        return self._bottom

    def _set_bottom(self, value):
        self._bottom = value
        self._centery = self._bottom - self._height // 2
        self._top = self._bottom - self._height

    def _del_bottom(self):
        del self._bottom

    bottom = property(
        fget=_get_bottom,
        fset=_set_bottom,
        fdel=_del_bottom,
        doc="The bottom property."
    )

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def move(self, x, y):
        self._set_centerx(self._centerx + x)
        self._set_centery(self._centery + y)
        return FakeRect(width=self._width,
                        height=self._height,
                        centerx=self._centerx,
                        centery=self._centery,
                        left=self._left,
                        right=self._right,
                        top=self._top,
                        bottom=self._bottom)


class PygameText:
    def __init__(self,
                 filename: str,
                 text: str,
                 text_size: int) -> None:
        self.filename = filename
        self.text = text
        self.text_size = text_size

        self.color = (255, 255, 255)  # White
        self.font = pygame.font.Font(self.filename, self.text_size)
        self.surface = self.font.render(self.text, True, self.color, None)
        self.rect = self.surface.get_rect()

    def set_position(self, left: int | float, top: int | float) -> None:
        self.rect.left, self.rect.top = left, top

    def get_position(self) -> tuple[int | float, int | float]:
        return self.rect.left, self.rect.top

    def set_text_color(self, RGB: tuple[int, int, int]) -> None:
        self.color = RGB
        self.surface = self.font.render(self.text, True, self.color, None)

    def get_text_color(self) -> tuple[int, int, int]:
        return self.color
