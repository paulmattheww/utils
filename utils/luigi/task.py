from luigi.local_target import LocalTarget
from dask.bytes.core import get_fs_token_paths

class Requirement:
    """Descriptor for Luigi Task requirement composition.
    Ths class implements the Python descriptor protocol. It is to be used together with
    the Requires descriptor class to compose Luigi Task requires().
    """
    def __init__(self, task_class, **params):
        """ Constructor of Requirement descriptor.
        The descriptor is to be instantiated during host class definition. Each instance
        assigned to a host class attribute represents a requirement that needs to satisfy
        before a Luigi Task can run. Each requirement is a Luigi Task (or its subclass).
        Args:
            task_class: Luigi Task or subclass, requirement for the host Task class
            params: kwargs to be passed to task_class during instantiation via task.clone()
        Returns:
            A Requirement descriptor instance.
        """
        self.task_class = task_class
        self.params = params

    def __get__(self, task, cls):
        """ Attribute access method of descriptor.
        When accessed, self.task_class is instantiated via task.clone() and returned as a
        Luigi Task instance.
        Args:
            task: class instance of host class when called via a host class' instance;
                None when called via the host class
            cls: host class object
        Returns:
            The Requirement descriptor instance when accessed via the host class;
            or self.task_class instance when accessed via a host class instance.
        """
        if task is None:
            return self
        return task.clone(self.task_class, **self.params)


class Requires:
    """Descriptor for Luigi Task requires() composition.
    Ths class implements the Python descriptor protocol. It is to be used together with
    the Requirement descriptor class to compose Luigi Task requires(). An instance of this
    descriptor should be assigned to the "requires" attribute of the host class during
    class definition.
    """
    def __get__(self, task, cls):
        """ Attribute access method of descriptor.
        Args:
            task: class instance of host class when called via a host class' instance;
                None when called via the host class
            cls: host class object
        Returns:
            The Requirement descriptor instance when accessed via the host class;
            or a function that implements "requires()" when accessed via a host class instance.
        """
        if task is None:
            return self
        return lambda: self(task)

    def __call__(self, task):
        """ Implements the "requires()" method of a Luigi Task.
        Together with Requirement, this method allows the descriptor to be used as "requires" composition
        of a Luigi Task.
        This method searches all attributes of the host class for instances of Requirement. Only these
        instances are returned.
        Args:
            task: host class instance
        Returns:
            A dictionary of Luigi tasks that are required before the task can run.
        """
        req_attr_list = [r for r in dir(task.__class__)
                         if isinstance(getattr(task.__class__, r), Requirement)]
        req_task_list = {i: getattr(task, i) for i in req_attr_list}
        return req_task_list


class TargetOutput:
    """ Descriptor for Luigi Task output composition.
    This class implements the Python descriptor protocol. This class can be used to compose Luigi
    Task output.
    """

    def __init__(self, file_pattern='{task.__class__.__name__}',
                 ext='.txt', target_class=LocalTarget, **target_kwargs):
        """ Constructor of TargetOutput descriptor.
        The descriptor is to be instantiated during host class definition. This descriptor should be
        assigned to the host class's "output" attribute.
        Args:
            file_pattern: str, a template to be evaluated when the "output" attribute is accessed
                via an instance of the host class; defaults to the name of the host task class.
            target_class: a Luigi Target class or subclass; defaults to LocalTarget.
            ext: str, the output file extension, defaults to ".txt"
                If there is a "glob" in target_kwargs, the extension is attached to the end of glob.
                Otherwise, the extension is attached to end of flie_pattern.
            target_kwargs: dictionary, named arguments to be passed to the target_class' constructor
        Returns:
            A TargetOutput descriptor instance.
        """
        self.target_class = target_class
        self.file_pattern = file_pattern
        self.ext = ext
        self.target_kwargs = target_kwargs


    def __get__(self, task, cls):
        """ Attribute access method of descriptor.
        Args:
            task: class instance of host class when called via a host class' instance;
                None when called via the host class
            cls: host class object
        Returns:
            The TargetOutput descriptor instance when accessed via the host class;
            or a function that implements "output()" when accessed via a host class instance.
        """
        if task is None:
            return self
        return lambda: self(task)

    def __call__(self, task):
        """ Implements the "output()" method of a Luigi Task.
        This method allows the descriptor to be used as "output" composition of a Luigi Task.
        A Target (or subsclass) is instantiated and returned. The target file path as well
        as the "file_pattern" template is evaluated here.
        Args:
            task: host class instance
        Returns:
            A Luigi Target (or subclass) instance.
        """

        # If there is a "glob" in target_kwargs, the extension is attached to the end of glob.
        # Otherwise, the extension is attached to end of flie_pattern.
        new_kwargs = {i:self.target_kwargs[i] for i in self.target_kwargs if i != 'glob' }

        if 'glob' in self.target_kwargs:
            target_path = self.file_pattern.format(task=task)

            new_glob = self.target_kwargs['glob'].format(task=task) + self.ext.format(task=task)
            new_kwargs['glob'] = new_glob
        else:
            target_path = (self.file_pattern.format(task=task) +
                           self.ext.format(task=task))

        # Make sure that the directory path ends with a system dependent separator.
        path_sep = get_fs_token_paths(target_path)[0].sep
        if target_path[-1] != path_sep:
            if target_path[-1] == "/":
                target_path = target_path[:-1]
            target_path = target_path + path_sep
        fs, _, _ = get_fs_token_paths(target_path)

        return self.target_class(target_path, **new_kwargs)
