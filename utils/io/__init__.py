
from atomicwrites import atomic_write as _backend_writer, AtomicWriter
from contextlib import contextmanager
import tempfile
import os
import io
import psutil

class SuffixWriter(AtomicWriter):

    def get_fileobject(self, suffix=None, prefix=tempfile.template, dir=None,
                       **kwargs):
        '''Return the temporary file to use ensuring suffix persists.'''
        if suffix is None:
            suffix = "." + str(self._path).split(".")[-1]
        elif not suffix.startswith("."):
            suffix = "." + suffix
        else:
            raise ValueError("Must specify a suffix as '.suffix'")
        if dir is None:
            dir = os.path.normpath(os.path.dirname(self._path))
        descriptor, name = tempfile.mkstemp(suffix=suffix, prefix=prefix,
                                            dir=dir)
        # io.open() will take either the descriptor or the name, but we need
        # the name later for commit()/replace_atomic() and couldn't find a way
        # to get the filename from the descriptor.
        os.close(descriptor)
        kwargs['mode'] = "w" #always a writer
        kwargs['file'] = name
        return io.open(**kwargs)


@contextmanager
def atomic_write(file, as_file=True, **cls_kwargs):
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

    # You can override things just fine...
    with _backend_writer(file, writer_cls=SuffixWriter, **cls_kwargs) as tmp:
        if as_file:
            yield tmp
        else:
            yield tmp.name


def memory():
    '''
    Measure memory usage; modified from:
    https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
    '''
    #w = WMI('.')
    #result = w.query("SELECT WorkingSet FROM Win32_PerfRawData_PerfProc_Process WHERE IDProcess=%d" % os.getpid())
    result = psutil.virtual_memory()[3]

    return result#int(result[0].WorkingSet)
