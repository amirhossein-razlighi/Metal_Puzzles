import mlx.core as mx
from utils import MetalKernel, MetalProblem


def map_spec(a: mx.array):
    return a + 10


def map_2D_test(a: mx.array):
    source = """
        uint thread_x = thread_position_in_grid.x;
        uint thread_y = thread_position_in_grid.y;
        uint linear_position = thread_x + thread_y * a_shape[1];
        if (thread_x < a_shape[0] && thread_y < a_shape[1]) {
          out[linear_position] = a[linear_position] + 10;
        }
    """

    kernel = MetalKernel(
        name="map_2D",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    return kernel


SIZE = 2
a = mx.arange(SIZE * SIZE).reshape((SIZE, SIZE))
output_shape = (SIZE, SIZE)

problem = MetalProblem(
    "Map 2D", map_2D_test, [a], output_shape, grid=(3, 3, 1), spec=map_spec
)
problem.show()
problem.check()