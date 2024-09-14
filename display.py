import random

import numpy as np

import graphicsUtils
import util
from Grid import Grid
from agents import Action

ROAD_COLOR = graphicsUtils.formatColor(0.2, 0.2, 0.2)
EDGE_COLOR = graphicsUtils.formatColor(1, 1, 1)
TEXT_COLOR = graphicsUtils.formatColor(1, 1, 1)
MUTED_TEXT_COLOR = graphicsUtils.formatColor(0.7, 0.7, 0.7)
LOCATION_COLOR = graphicsUtils.formatColor(1, 1, 1)
HOME_COLOR = graphicsUtils.formatColor(0.13, 0.55, 0.13)
OBSTACLE_COLOR = HOME_COLOR
TAKEN_PARKING = graphicsUtils.formatColor(255 / 255, 100 / 255, 100 / 255)
AVAILABLE_PARKING = graphicsUtils.formatColor(144 / 255, 238 / 255, 144 / 255)
BACKGROUND_COLOR = graphicsUtils.formatColor(0, 0, 0)


class Display:

    def __init__(self, parking_map: Grid, size=50, min_val=0, max_val=.15):
        self.size = size
        self.margin = self.size * 0.75
        self.parking_map = parking_map
        self.images = {}
        self.rand_float = {(x, y): random.random() for x in range(parking_map.width) for y in range(parking_map.height)}
        self.min_val = min_val
        self.max_val = max_val

    def setup(self):
        global SCREEN_WIDTH, SCREEN_HEIGHT, GRID_HEIGHT
        screen_width = (self.parking_map.width - 1) * self.size + self.margin * 2
        screen_height = (self.parking_map.height - 0.5) * self.size + self.margin * 2

        graphicsUtils.begin_graphics(screen_width,
                                     screen_height,
                                     graphicsUtils.formatColor(0, 0, 0), title="Parking Finder")
        self.load_images()

    def load_images(self):
        self.images['car'] = "images/car.png"
        self.images['home'] = "images/home.png"
        self.images['road'] = "images/road.png"
        self.images['parking cars'] = [f"images/parking_car{i}.png" for i in range(1, 5)]
        self.images['houses'] = [f"images/house{i}.png" for i in range(1, 5)]
        self.images['trees'] = [f"images/tree{i}.png" for i in range(1, 5)]

    def drawGrid(self, grid, isFinished=False, action=None, isDiaplayRun=True):
        graphicsUtils.clear_screen()
        min_val = self.min_val if isDiaplayRun else min(
            [cell for row in grid for cell in row if isinstance(cell, (int, float))], default=0)
        max_val = self.max_val if isDiaplayRun else max(
            [cell for row in grid for cell in row if isinstance(cell, (int, float))], default=1)

        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                val = -1
                isObstacle = False
                isHome = False
                isCurrent = False
                isRoad = False

                if isinstance(cell, (int, float)):
                    val = cell
                elif cell == "#":
                    isObstacle = True
                elif cell == "c":
                    isCurrent = True
                elif cell == "e":
                    isHome = True
                elif cell == " ":
                    isRoad = True

                self.drawSquare(x, y, val, min_val, max_val, isRoad, isObstacle, isCurrent, isHome, isFinished, action,
                                isDiaplayRun)

    def drawSquare(self, x, y, val, min, max, isRoad, isObstacle, isCurrent, isHome, isFinished, action, isDiaplayRun):
        square_color = self.getColor(val, max, min)
        (screen_x, screen_y) = self.to_screen((x, y))
        image_path = None
        angle = 0
        isParking = True
        rand_float = self.rand_float[(x, y)]
        if isObstacle:
            isParking = False
            square_color = OBSTACLE_COLOR
            image_path = self.images["houses"][
                round(rand_float * (1 / 0.8) * (len(self.images["houses"]) - 1))] if rand_float < 0.8 \
                else self.images["trees"][round((rand_float - 0.8) * (1 / 0.2) * (len(self.images["trees"]) - 1))]
        if isHome:
            isParking = False
            square_color = HOME_COLOR
            image_path = self.images["home"]
        if isRoad:
            isParking = False
            square_color = ROAD_COLOR
        if val == 0:
            if isDiaplayRun:
                square_color = TAKEN_PARKING
                image_path = self.images["parking cars"][round(rand_float * (len(self.images["parking cars"]) - 1))]
            else:
                square_color = BACKGROUND_COLOR
        if val == 1:
            square_color = AVAILABLE_PARKING
        if isCurrent:
            isParking = False
            square_color = AVAILABLE_PARKING if isFinished else ROAD_COLOR
            image_path = self.images["car"]
            if action == Action.DOWN:
                angle = 180
            if action == Action.RIGHT:
                angle = 90
            if action in (Action.LEFT, Action.EXIT):
                angle = 270

        graphicsUtils.square((screen_x, screen_y), 0.5 * self.size, color=square_color, filled=1)
        if isParking:
            graphicsUtils.square((screen_x, screen_y), 0.5 * self.size, color=EDGE_COLOR, filled=0)
        if image_path:
            graphicsUtils.image((screen_x - 0.5 * self.size, screen_y - 0.5 * self.size), image_path,
                                (self.size, self.size), angle)

    def getColor(self, value, maxVal, minVal):
        """
        Returns a proportional shade of light purple based on the input value.

        Parameters:
        value (float): A number between 0 and 1, where 0 is the lightest shade and 1 is the darkest.

        Returns:
        str: A hexadecimal color code representing the proportional light purple shade.
        """
        value = (value - minVal) / (maxVal - minVal)
        value = (value if value < 1 else 1) if value > 0 else 0
        # Define the RGB values for the lightest and darkest shades of light purple
        lightest = (230, 230, 255)  # Lightest shade of light purple
        darkest = (150, 120, 190)  # Darkest shade of light purple

        # Calculate the proportional RGB values
        r = int(lightest[0] * (1 - value) + darkest[0] * value)
        g = int(lightest[1] * (1 - value) + darkest[1] * value)
        b = int(lightest[2] * (1 - value) + darkest[2] * value)

        # Return the color in hexadecimal format
        return graphicsUtils.formatColor(r / 255.0, g / 255.0, b / 255.0)

    def to_screen(self, point):
        (gamex, gamey) = point
        x = gamex * self.size + self.margin
        y = (self.parking_map.height - gamey - 1) * self.size + self.margin
        return (x, y)
