from random import randrange
from contextlib import contextmanager
import io

from luigi import Task, ExternalTask
from luigi.local_target import LocalTarget, atomic_file
from luigi.format import FileWrapper, Nop

class suffix_preserving_atomic_file(atomic_file):
    '''Wraps Luigi's atomic_file(AtomicLocalFile) object.
    Ensures the suffix is preserved.
    '''
    def generate_tmp_path(self, path):
        if '.' in path:
            suffix = str(path).split('.')[-1]
            path_presuffix = str(path).replace('.' + suffix, '')
        else:
            raise ValueError("Must have valid file suffix.")
        return path_presuffix + f"-luigi-tmp-{randrange(0, 1e10)}.{suffix}"


class BaseAtomicProviderLocalTarget(LocalTarget):
    def __init__(self, path, atomic_provider=atomic_file, is_tmp=False, format=Nop):
        # Allow some composability of atomic handling
        self.path = path
        self.atomic_provider = atomic_provider
        self.is_tmp = is_tmp
        self.format = format

    def open(self, mode='r'):
        '''Modifying any code in LocalTarget to use self.atomic_provider
        rather than atomic_file
        '''
        rwmode = mode.replace('b', '').replace('t', '')
        if rwmode == 'w':
            self.makedirs()
            return self.format.pipe_writer(self.atomic_provider(self.path))
        elif rwmode == 'r':
            fileobj = FileWrapper(io.BufferedReader(io.FileIO(self.path, mode)))
            return self.format.pipe_reader(fileobj)

        else:
            raise Exception(f"The kwarg `mode` must be 'r' or 'w' (got: `{mode}`)")

    @contextmanager
    def temporary_path(self):
        # NB: unclear why LocalTarget doesn't use atomic_file in its implementation
        self.makedirs()
        with self.atomic_provider(self.path) as af:
            yield af.tmp_path


class SuffixPreservingLocalTarget(BaseAtomicProviderLocalTarget):
    def __init__(self, path):
        super().__init__(path, atomic_provider=suffix_preserving_atomic_file, is_tmp=False, format=Nop)
