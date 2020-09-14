from contextlib import contextmanager
from torch import distributed


@contextmanager
def zero_first():
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if distributed.get_rank() != 0:
        distributed.barrier()
    yield
    if distributed.get_rank() == 0:
        distributed.barrier()


@contextmanager
def sequence():
    for _ in range(distributed.get_rank()):
        distributed.barrier()
    yield
    for _ in range(distributed.get_world_size() - distributed.get_rank()):
        distributed.barrier()
