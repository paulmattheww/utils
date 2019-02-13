#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `utils` package."""

import os
from tempfile import TemporaryDirectory
from unittest import TestCase

from pandas import read_excel, read_parquet

from utils.hash_str import hash_str, get_csci_salt, get_user_id
from utils.io import atomic_write

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

    def test_parquet_atomic_write(self):
        """Ensure parquet file exists after being written successfully"""
        df = read_excel("data/hashed.xlsx")
        first_5_xlsx = df["hashed_id"].head().tolist()

        with TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, 'test.parquet')
            assert not os.path.exists(fp)

            with atomic_write(fp, ext=".parquet") as f:
                fname = f.name
                df.to_parquet(fname)

            assert os.path.exists(fp)

            df = read_parquet(fp)
            first_5_parquet = df["hashed_id"].head().tolist()

            assert first_5_parquet == first_5_xlsx

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
