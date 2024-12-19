import mlx.core as mx
from utils import MetalProblem, MetalKernel


def map_spec(a: mx.array):
    return a + 10


def map_threadgroup_2D_test(a: mx.array):
    source = """
        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        uint j = threadgroup_position_in_grid.y * threads_per_threadgroup.y + thread_position_in_threadgroup.y;
        
        if (i < a_shape[0] && j < a_shape[1]) {
            out[i * a_shape[1] + j] = a[i * a_shape[1] +
                j] + 10;
        }
    """

    kernel = MetalKernel(
        name="threadgroups_2D",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    return kernel


SIZE = 5
a = mx.ones((SIZE, SIZE))
output_shape = (SIZE, SIZE)

problem = MetalProblem(
    "Threadgroups 2D",
    map_threadgroup_2D_test,
    [a],
    output_shape,
    grid=(6, 6, 1),
    threadgroup=(3, 3, 1),
    spec=map_spec,
)
problem.show()
problem.check()
