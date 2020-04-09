from contextlib import contextmanager
from itertools import cycle

@contextmanager
def get_gpu_id():
    allocated_gid = None
    for gid in cycle([1, 2, 3, 4]):
        if os.system(f"lockfile /var/lock/LCK_gpu_{gid}.lock") == 0: #locked
            allocated_gid = gid
            break
    yield allocated_gid
    os.system(f"rm -f /var/lock/LCK_gpu_{allocated_gid}.lock")
