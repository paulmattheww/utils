import collections

def compare_two_sets(i1, i2, output=False):
    '''i1 and i2 can have repeated values.'''
    s1 = set(i1)
    s2 = set(i2)
    print('Length of set1: ', len(s1))
    print('Length of set2: ', len(s2))
    intersection = s1.intersection(s2)
    print('Number of common elements: ', len(intersection))
    if output:
        return intersection

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
