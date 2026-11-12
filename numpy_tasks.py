"""Заготовки задач на NumPy."""
from typing import Any

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    n = matrices[0].shape[0]
    result = np.zeros((n,1))
    for i in range(len(matrices)):
        result += matrices[i] @ vectors[i]
    return result


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return np.where(matrix > threshold, 1, 0)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    result = []
    for row in matrix:
        result.append(np.unique(row).tolist())
    return result



def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    result = []
    for j in range(matrix.shape[1]):
        result.append(np.unique(matrix[:, j]).tolist())
    return result


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(mean, std, (rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )

def plot_histogram(stats: MatrixStatistics) -> None:
    import matplotlib.pyplot as plt

    plt.subplot(2, 2, 1)
    plt.hist(stats.row_means)
    plt.title("Средние по строкам")

    plt.subplot(2, 2, 2)
    plt.hist(stats.column_means)
    plt.title("Средние по стролбцам")

    plt.subplot(2, 2, 3)
    plt.hist(stats.row_variances)
    plt.title('Дисперсии по строкам')

    plt.subplot(2, 2, 4)
    plt.hist(stats.column_variances)
    plt.title("Дисперсии по столбцам")

    plt.tight_layout()
    plt.show()



def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    result = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            if (i + j) % 2 == 0:
                result[i, j] = first
            else:
                result[i, j] = second
    return result


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = background_color

    top = (image_height - height) // 2
    left = (image_width - width) // 2
    image[top:top + height, left:left + width] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    a, b = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = background_color

    x0 = (image_width - 1) / 2
    y0 = (image_height - 1) / 2
    for y in range(image_height):
        for x in range(image_width):
            if (x - x0) ** 2 / a ** 2 + (y - y0) ** 2 / b ** 2 <= 1:
                image[y, x] = shape_color
    return image

def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    window = data.window

    mean = np.mean(values)
    variance = np.var(values)
    std = np.std(values)

    maxima = []
    minima = []

    for i in range(1, len(values) -1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            maxima.append(i)
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            minima.append(i)

    moving_average = []
    for i in range(len(values) - window + 1):
        moving_average.append(np.mean(values[i:i + window]))

    return TimeSeriesStatistics(
        mean=float(mean),
        variance=float(variance),
        std=float(std),
        local_maxima_indices=np.array(maxima, dtype=int),
        local_minima_indices=np.array(minima, dtype=int),
        moving_average=np.array(moving_average),
    )

def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = np.asarray(data.labels), data.class_count
    if class_count is None:
        class_count = int(labels.max()) + 1
    result = np.zeros((len(labels), class_count), dtype=int)
    for i in range(len(labels)):
        result[i, labels[i]] = 1
    return result


#Тесты

RED = (255, 0, 0)
BLACK = (0, 0, 0)


def _close(a, b):
    return abs(a - b) < 1e-9


def test_sum_prod():
    matrices = np.array([np.eye(2), 2 * np.eye(2)])
    vectors = np.array([[[1], [2]], [[3], [4]]])
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    assert result.shape == (2, 1)
    assert np.array_equal(result, [[7], [10]])  # [1,2] + [6,8]


def test_sum_prod_single_pair():
    matrices = np.array([[[1, 2], [3, 4]]])
    vectors = np.array([[[1], [1]]])
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    assert np.array_equal(result, [[3], [7]])


def test_binarize():
    matrix = np.array([[0.2, 0.5], [0.7, 1.0]])
    result = binarize(BinarizeInput(matrix, 0.5))
    assert np.array_equal(result, [[0, 0], [1, 1]])  # 0.5 не строго больше
    assert matrix[0, 0] == 0.2                       # исходная матрица не изменилась


def test_unique_rows_and_columns():
    assert unique_rows(MatrixInput(np.array([[1, 1, 2], [3, 3, 3]]))) == [[1, 2], [3]]
    assert unique_columns(MatrixInput(np.array([[1, 2], [1, 3]]))) == [[1], [2, 3]]


def test_matrix_statistics():
    stats = matrix_statistics(RandomMatrixInput(rows=3, columns=4, mean=0, std=1, seed=1))
    assert stats.matrix.shape == (3, 4)
    assert stats.row_means.shape == (3,)
    assert stats.column_means.shape == (4,)
    assert np.allclose(stats.row_means, stats.matrix.mean(axis=1))
    assert np.allclose(stats.column_variances, stats.matrix.var(axis=0))


def test_matrix_statistics_seed():
    first = matrix_statistics(RandomMatrixInput(rows=2, columns=2, seed=5))
    second = matrix_statistics(RandomMatrixInput(rows=2, columns=2, seed=5))
    assert np.array_equal(first.matrix, second.matrix)  # одинаковый seed


def test_matrix_statistics_zero_std():
    stats = matrix_statistics(RandomMatrixInput(rows=3, columns=2, mean=5, std=0, seed=1))
    assert np.allclose(stats.matrix, 5)
    assert np.allclose(stats.row_variances, 0)


def test_chess():
    result = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    assert np.array_equal(result, [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    assert np.array_equal(chess(ChessInput(1, 1, 7, 9)), [[7]])  # левый верхний = first


def test_draw_rectangle():
    image = draw_rectangle(RectangleInput(
        width=3, height=1, image_height=5, image_width=5,
        shape_color=RED, background_color=BLACK))
    assert image.shape == (5, 5, 3)
    assert tuple(image[2, 1]) == RED and tuple(image[2, 3]) == RED
    assert tuple(image[2, 0]) == BLACK and tuple(image[0, 0]) == BLACK
    assert np.all(image == RED, axis=2).sum() == 3  # всего 3 красных пикселя


def test_draw_rectangle_full_image():
    image = draw_rectangle(RectangleInput(
        width=4, height=2, image_height=2, image_width=4,
        shape_color=RED, background_color=BLACK))
    assert np.all(image == RED, axis=2).all()  # прямоугольник закрыл всё


def test_draw_ellipse():
    image = draw_ellipse(EllipseInput(
        semi_axis_x=2, semi_axis_y=2, image_height=5, image_width=5,
        shape_color=RED, background_color=BLACK))
    assert image.shape == (5, 5, 3)
    assert tuple(image[2, 2]) == RED   # центр
    assert tuple(image[2, 0]) == RED   # на границе (<= 1)
    assert tuple(image[0, 0]) == BLACK # угол снаружи


def test_draw_ellipse_small():
    image = draw_ellipse(EllipseInput(
        semi_axis_x=1, semi_axis_y=1, image_height=5, image_width=5,
        shape_color=RED, background_color=BLACK))
    assert np.all(image == RED, axis=2).sum() == 5  # центр и 4 соседа


def test_analyze_time_series():
    result = analyze_time_series(TimeSeriesInput(values=[1, 3, 2, 5, 4], window=2))
    assert _close(result.mean, 3)
    assert _close(result.variance, 2)
    assert _close(result.std, 2 ** 0.5)
    assert list(result.local_maxima_indices) == [1, 3]
    assert list(result.local_minima_indices) == [2]
    assert np.allclose(result.moving_average, [2, 2.5, 3.5, 4.5])
    assert len(result.moving_average) == 5 - 2 + 1


def test_analyze_time_series_edge_cases():
    increasing = analyze_time_series(TimeSeriesInput(values=[1, 2, 3], window=1))
    assert len(increasing.local_maxima_indices) == 0   # края не считаются
    assert len(increasing.local_minima_indices) == 0
    assert np.allclose(increasing.moving_average, [1, 2, 3])  # окно 1

    plateau = analyze_time_series(TimeSeriesInput(values=[1, 2, 2, 1], window=2))
    assert len(plateau.local_maxima_indices) == 0      # максимум должен быть строгим


def test_one_hot():
    result = one_hot(OneHotInput(labels=[0, 2, 3, 0]))
    assert np.array_equal(result, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])


def test_one_hot_class_count():
    assert one_hot(OneHotInput(labels=[0, 2], class_count=5)).shape == (2, 5)
    assert np.array_equal(one_hot(OneHotInput(labels=[0, 0])), [[1], [1]])
    assert np.array_equal(one_hot(OneHotInput(labels=[1])), [[0, 1]])




























