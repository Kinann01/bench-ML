#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from plotter import Plotter
from util import DatasetUtil

ROOT = Path(__file__).resolve().parent.parent


"""

"""


class Analyzer:

    def __init__(self, plotter : Plotter):
        self.plotter = plotter

    def set_context(self, util_object : DatasetUtil):
        self.util_object = util_object

    def analyze(self):
        pass
