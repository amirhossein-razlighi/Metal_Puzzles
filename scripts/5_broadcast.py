import mlx.core as mx
from utils import MetalKernel, MetalProblem


def zip_spec(a: mx.array, b: mx.array):
    return a + b


def broadcast_test(a: mx.array, b: mx.array):
    source = """
        uint thread_x = thread_position_in_grid.x;
        uint thread_y = thread_position_in_grid.y;
        uint broadcasted_size_x = max(a_shape[0], b_shape[0]);
        uint broadcasted_size_y = max(a_shape[1], b_shape[1]);

        if (thread_x < broadcasted_size_x && thread_y < broadcasted_size_y) {
            out[thread_x * broadcasted_size_x + thread_y] = a[0 + thread_y] + b[thread_x + 0];
        }
    """

    kernel = MetalKernel(
        name="broadcast",
        input_names=["a", "b"],
        output_names=["out"],
        source=source,
    )

    return kernel


SIZE = 2
a = mx.arange(SIZE).reshape(SIZE, 1)
b = mx.arange(SIZE).reshape(1, SIZE)
output_shape = (SIZE, SIZE)

problem = MetalProblem(
    "Broadcast", broadcast_test, [a, b], output_shape, grid=(3, 3, 1), spec=zip_spec
)
problem.show()
problem.check()