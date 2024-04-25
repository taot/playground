import numpy as np
import random
from matplotlib import pyplot as plt


segment_colors = [
    0xC0C0C0,   # silver
    0x808080,   # gray
    0xFF0000,   # red
    0x800000,   # maroon
    0xFFFF00,   # yellow
    0x808000,   # olive
    0x00FF00,   # lime
    0x008000,   # green
    0x00FFFF,   # aqua
    0x008080,   # teal
    0x0000FF,   # blue
    0x000080,   # navy
    0xFF00FF,   # fuchsia
    0x800080,   # purple
]


def get_colored_segmentation(segments):
    if len(segments.shape) != 2:
        raise Exception("segment shape invalid: " + str(segments.shape))
    graph = __create_segmentation_graph(segments)
    colors = np.zeros((graph.shape[0]), dtype=long)
    colors[0] = segment_colors[0]
    size = segments.max() + 1
    for i in range(1, size):
        neighbor_colors = []
        for j in range(0, size):
            if graph[i][j] == 1 and colors[j] > 0 and colors[j] not in neighbor_colors:
                neighbor_colors.append(colors[j])
        while True:
            k = random.randint(0, len(segment_colors) - 1)
            if segment_colors[k] not in neighbor_colors:
                colors[i] = segment_colors[k]
                break
    nrows, ncols = segments.shape
    colored_segments = np.zeros((nrows, ncols, 3), dtype=np.uint8)
    for i in range(0, nrows):
        for j in range(0, ncols):
            seg = segments[i][j]
            colored_segments[i][j] = __get_rgb_tuple(colors[seg])

    return colored_segments


def __get_rgb_tuple(hex):
    return hex / 0x010000, (hex % 0x010000) / 0x000100, hex % 0x000100


def __create_segmentation_graph(segments):
    size = segments.max() + 1
    graph = np.zeros((size, size))
    nrows, ncols = segments.shape
    for i in range(0, nrows-1):
        for j in range(0, ncols-1):
            n0 = segments[i][j]
            n1 = segments[i][j+1]
            n2 = segments[i+1][j]
            n3 = segments[i+1][j+1]
            if n0 != n1:
                graph[n0][n1] = graph[n1][n0] = 1
            if n0 != n2:
                graph[n0][n2] = graph[n2][n0] = 1
            if n0 != n3:
                graph[n0][n3] = graph[n3][n0] = 1

    return graph
