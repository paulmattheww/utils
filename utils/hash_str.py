# Import statements
import hashlib
import os

def get_csci_salt():
    """Returns the appropriate salt for CSCI E-29

    :rtype: bytes
    """
    salt = os.environ["CSCI_SALT"].encode()
    return salt


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


def get_user_id(username):
    """Fetches last 8 of hash_str(username) with salt
    specified from .env file.

    :param str username:  username in question
    """
    salt = get_csci_salt().decode("utf-8")
    return hash_str(username.lower(), salt=salt).hex()[:8]
