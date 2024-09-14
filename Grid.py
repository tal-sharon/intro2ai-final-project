class Grid:
    """
    A 2-dimensional array of immutables backed by a list of lists.  Data is accessed
    via grid[x][y] where (x,y) are cartesian coordinates with x horizontal,
    y vertical and the origin (0,0) in the bottom left corner.

    The __str__ method constructs an output that is oriented appropriately.
    """
    def __init__(self, width, height, initialValue=' '):
        self.width = width
        self.height = height
        self.data = [[initialValue for y in range(height)] for x in range(width)]
        self.terminalState = 'TERMINAL_STATE'

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __eq__(self, other):
        if other == None: return False
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def copy(self):
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def _getLegacyText(self):
        t = [[self.data[x][y] for x in range(self.width)] for y in range(self.height)]
        t.reverse()
        return t

    def __str__(self):
        return str(self._getLegacyText())


def makeGrid(gridString):
  width, height = len(gridString[0]), len(gridString)
  grid = Grid(width, height)
  for ybar, line in enumerate(gridString):
    y = height - ybar - 1
    for x, el in enumerate(line):
      grid[x][y] = el
  return grid