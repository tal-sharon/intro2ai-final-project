class GameState:
    def __init__(self, row, col, l: bool, r: bool, u: bool, d: bool):
        self.row = row
        self.col = col
        self.l, self.r, self.u, self.d = l, r, u, d

    def __eq__(self, other):
        return (self.row == other.row and self.col == other.col and
                self.l == other.l and self.r == other.r and self.u == other.u and self.d == other.d)

    def __hash__(self):
        return hash(self.row) + hash(self.col) + hash(self.l) + hash(self.r) + hash(self.u) + hash(self.d)

    def __str__(self):
        return str((self.row, self.col, self.l, self.r, self.u, self.d))
