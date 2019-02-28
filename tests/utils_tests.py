#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `utils` package."""

import os
from tempfile import TemporaryDirectory
from unittest import TestCase

from pandas import read_excel, read_parquet
import numpy as np

from utils.text.dlp import hash_str
from utils.io import atomic_write
from utils.ml.metrics import multiclass_confusion_matrix, binary_confusion_matrix


class TestMlModule(TestCase):
     def test_multiclass_confusion_matric(self):
         cm = None
         try:
             cm = multiclass_confusion_matrix(np.random.randint(0,5,20), np.random.randint(0,5,20))
         self.assertEqual()

class FakeFileFailure(IOError):
    pass


class HashTests(TestCase):

    def test_basic(self):
        self.assertEqual(
            hash_str('world!', salt='hello, ')[:6], b'68e656')

    def test_get_salt(self):
        self.assertEqual(get_csci_salt()[:8],  b'3f87b3a5')

    def test_get_user_id(self):
        self.assertEqual(get_user_id("thisispawbz"), '34616539')

    def test_no_nacl(self):
        self.assertEqual(
            hash_str('this is NOT salted!')[:6], b'1e1a41')


class TestAtomics(TestCase):

    def test_writer(self):
        with TemporaryDirectory() as tmpdir:
            file = os.path.join(tmpdir, 'asdf.txt')
            assert not os.path.isfile(file)
            with atomic_write(file) as f:
                f.write('asdf')
                assert f.name.endswith('.txt')
            assert os.path.isfile(file)

            with open(file) as f:
                assert f.read() == 'asdf'

    def test_atomic_fail(self):
        with TemporaryDirectory() as tmpdir:
            file = os.path.join(tmpdir, 'asdf.txt')
            assert not os.path.exists(file)
            with atomic_write(file, as_file=False) as fhandle:
                with open(fhandle, "w") as f:
                    f.write('asdf')
                    assert str(file).endswith('.txt')
            assert os.path.exists(file)


class AtomicWriteTests(TestCase):

    def test_atomic_write(self):
        """Ensure file exists after being written successfully"""

        with TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, 'asdf.txt')

            with atomic_write(fp, 'w') as f:
                assert not os.path.exists(fp)
                tmpfile = f.name
                f.write('asdf')

            assert not os.path.exists(tmpfile)
            assert os.path.exists(fp)

            with open(fp) as f:
                self.assertEqual(f.read(), 'asdf')

    # def test_parquet_atomic_write(self):
    #     """Ensure parquet file exists after being written successfully"""
    #     df = read_excel("data/hashed.xlsx")
    #     first_5_xlsx = df["hashed_id"].head().tolist()
    #
    #     with TemporaryDirectory() as tmp:
    #         fp = os.path.join(tmp, 'test.parquet')
    #         assert not os.path.exists(fp)
    #
    #         with atomic_write(fp, ext=".parquet") as f:
    #             fname = f.name
    #             df.to_parquet(fname)
    #
    #         assert os.path.exists(fp)
    #
    #         df = read_parquet(fp)
    #         first_5_parquet = df["hashed_id"].head().tolist()
    #
    #         assert first_5_parquet == first_5_xlsx

    def test_atomic_failure(self):
        """Ensure that file does not exist after failure during write"""

        with TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, 'asdf.txt')

            with self.assertRaises(FakeFileFailure):
                with atomic_write(fp, 'w') as f:
                    tmpfile = f.name
                    assert os.path.exists(tmpfile)
                    raise FakeFileFailure()

        assert not os.path.exists(tmpfile)
        assert not os.path.exists(fp)

    def test_file_exists(self):
        """Ensure an error is raised when a file exists"""
        with TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, 'file_exists_test.txt')

            with atomic_write(fp, 'w') as f:
                assert not os.path.exists(fp), FileExistsError("FileExistsError")
                f.write('Testing if file exists.')
