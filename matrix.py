import json
import random


class Matrix:
    class MatrixException(Exception):
        pass

    @classmethod
    def from_array(cls, array):
        rows = len(array)
        try:
            cols = len(array[0])
        except TypeError:
            cols = 1
        return cls(rows, cols, values=array)

    def __init__(self, rows, cols, values=None):
        self.rows = rows
        self.cols = cols
        self.matrix = self.validate_values(values) if values is not None else self.create_emptry_matrix()
        self.shape = self.rows, self.cols

    def __repr__(self):
        return json.dumps(self.matrix)

    def validate_values(self, values):
        if len(values) != self.rows or any([len(cols) != self.cols for cols in [row for row in values]]):
            raise self.MatrixException('Values must be the same shape declared on the Matrix to be assigned')
        return values

    def show_table(self):
        print(self.get_table())

    def get_table(self):
        body = ['|'.join(['{:4}'.format(item) for item in row]) for row in self.matrix]
        line_separator = "\n+" + (('-' * 4) + '+') * self.cols + '\n'
        table = line_separator
        for row in body:
            table += '|' + row + '|'
            table += line_separator
        return table

    def create_emptry_matrix(self, rows=None, cols=None):
        rows = self.rows if rows is None else rows
        cols = self.cols if cols is None else cols
        return [[0 for _ in range(cols)] for _ in range(rows)]

    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = random.randint(1, 10)

    def transpose(self):
        values = [[self.matrix[j][i] for j in range(self.rows)] for i in range(self.cols)]
        new_matrix = Matrix(self.cols, self.rows, values=values)
        return new_matrix

    def map(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = func(self.matrix[i][j])

    def __add__(self, other):
        new_matrix = self.create_emptry_matrix()
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise self.MatrixException('Matrices must be the same shape to add them')
            for i in range(self.rows):
                for j in range(self.cols):
                    new_matrix[i][j] = self.matrix[i][j] + other.matrix[i][j]
        elif type(other) in [int, float]:
            for i in range(self.rows):
                for j in range(self.cols):
                    new_matrix[i][j] = self.matrix[i][j] + other
        else:
            raise self.MatrixException("Matrices can only be added to ints, floats, or other matrices")
        return new_matrix

    def __sub__(self, other):
        new_matrix = self.create_emptry_matrix()
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise self.MatrixException('Matrices must be the same shape to subtract them')
            for i in range(self.rows):
                for j in range(self.cols):
                    new_matrix[i][j] = self.matrix[i][j] - other.matrix[i][j]
        elif type(other) in [int, float]:
            for i in range(self.rows):
                for j in range(self.cols):
                    new_matrix[i][j] = self.matrix[i][j] - other
        else:
            raise self.MatrixException("Matrices can only be subtracted with ints, floats, or other matrices")
        return new_matrix

    def __mul__(self, other):
        if type(other) == list:
            return [sum(row[i] * other[i] for i in range(self.cols)) for row in self.matrix]
        # if self.rows != other.cols:
        #     raise self.MatrixException('Incompatible matrix shapes')
        new_matrix = self.create_emptry_matrix(rows=self.rows, cols=other.cols)
        for i in range(len(new_matrix)):
            for j in range(len(new_matrix[0])):
                new_matrix[i][j] = sum(self.matrix[i][n] * other.matrix[n][j] for n in range(self.cols))
        return new_matrix
