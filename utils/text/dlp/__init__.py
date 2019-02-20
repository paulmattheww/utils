# Import statements
import hashlib
import os


def hash_str(some_val, salt=''):
    """Converts strings to hash digest

    See: https://en.wikipedia.org/wiki/Salt_(cryptography)

    :param str or bytes some_val: thing to hash

    :param str or bytes salt: string or bytes to add randomness to the hashing,
        defaults to ''.

    :rtype: bytes
    """
    # handle some_val as str or bytes
    if type(some_val) == str:
        some_val_bytes = some_val.encode()
    elif type(some_val) == bytes:
        some_val_bytes = some_val
    else:
        raise TypeError("TypeError")

    # handle salt as str or bytes
    if type(salt) == str:
        salt_bytes = salt.encode()
    elif type(salt) == bytes:
        salt_bytes = salt
    else:
        raise TypeError("TypeError")

    # hash salt first then value, return bytes
    m = hashlib.sha256(salt_bytes)
    m.update(some_val_bytes)
    return m.hexdigest().encode()
