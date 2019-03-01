# Import statements
import hashlib
import os
import re

def basic_dlp_str(text):
    """
    DESCR: Take a single value and send it to the api
    INPUT: text - str - piece of text to apply dlp to
    OUTPUT: str - lots of stuff, only take results
    """
    assert isinstance(text, str), TypeError("Must pass <str>!")
    # First pass at ssn and phone to keep API costs down
    # as well as eliminate items we know the API misses
    # basic regex ssn and phone formats
    re_dict = dict(basic_ssn_format = [r"\d{3}-\d{2}-\d{4}", "***-**-****"],
                   basic_ssn_nodashes_format = [r"\d{9}", "*********"],
                   basic_ssn_per_format = [r"\d{3}.\d{2}.\d{4}", "***.**.****"],
                   basic_tel10_format = [r"\d{3}-\d{3}-\d{4}", "***-***-****"],
                   basic_tel10_par_format = [r"\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}", "(***) ***-****"],
                   basic_tel10_dot_format = [r"\d{3}.\d{3}.\d{4}", "***.***.****"],
                   basic_tel10_nodashes_format = [r"\d{10}", "**********"],
                   basic_tel7_format = [r"\d{3}-\d{4}", "***-****"],
                   basic_tel7_nodashes_format = [r"\d{7}", "*******"],
                   )
    for k, criteria in re_dict.items():
        pattern = re.compile(criteria[0])
        text = re.sub(pattern, criteria[1], text)
    return text



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
