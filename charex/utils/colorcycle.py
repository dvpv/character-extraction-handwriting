from typing import List

DEFAULT_COLORS = [
    (255, 0, 0),  # #F00
    (0, 255, 0),  # #0F0
    (0, 0, 255),  # #00F
    (255, 255, 0),  # #FF0
    (255, 0, 255),  # #F0F
    (0, 255, 255),  # #0FF
]


class ColorCycle:
    __colors: List[tuple] = DEFAULT_COLORS
    __index: int = 0

    def __init__(self, colors: List[tuple] = None):
        if colors != None and colors != [] and colors is list:
            self.__colors = colors

    def next(self) -> tuple:
        color = self.__colors[self.__index]
        self.__index += 1
        if self.__index == len(self.__colors):
            self.__index = 0
        return color
