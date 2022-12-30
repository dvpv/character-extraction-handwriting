from charex.utils.colorcycle import ColorCycle, DEFAULT_COLORS
from unittest import TestCase


class ColorCycleTest(TestCase):
    def test_cycle_default(self):
        cc = ColorCycle()
        index = 0
        assert cc.next() == DEFAULT_COLORS[index]
        index += 1
        while index != 0:
            assert cc.next() == DEFAULT_COLORS[index]
            index += 1
            if index == len(DEFAULT_COLORS):
                index = 0

    def test_cycle_custom(self):
        cc = ColorCycle(
            colors=[
                (255,128,0), # #f80
                (128,128,0), # #880
                (0,0,128), # #008
                (128,255,255), # #8ff
            ],
        )
        index = 0
        assert cc.next() == DEFAULT_COLORS[index]
        index += 1
        while index != 0:
            assert cc.next() == DEFAULT_COLORS[index]
            index += 1
            if index == len(DEFAULT_COLORS):
                index = 0
