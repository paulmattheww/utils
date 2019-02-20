from contextlib import contextmanager
import uuid
import os
import shutil
from pathlib import Path


@contextmanager
def atomic_write(file, mode='w', as_file=True, ext=".txt", **kwargs):
    """Write a file atomically

    :param file: str or :class:`os.PathLike` target to write

    :param bool as_file:  if True, the yielded object is a :class:File.
        (eg, what you get with `open(...)`).  Otherwise, it will be the
        temporary file path string

    :param kwargs: anything else needed to open the file

    :raises: FileExistsError if target exists

    Example::

        with atomic_write("hello.txt") as f:
            f.write("world!")

    """
    # make os.path objects for temporary file and target file
    tmpfile = os.path.join(os.getcwd(), "tmp", str(uuid.uuid4()) + ext)
    target_file = os.path.join(os.getcwd(), file)

    # create empty file then open context
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    Path(tmpfile).touch()
    tmp = open(tmpfile, mode, **kwargs)

    try:
        if as_file:
            yield tmp
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
        else:
            yield str(tmp.name)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
    finally:
        shutil.move(tmpfile, target_file)
