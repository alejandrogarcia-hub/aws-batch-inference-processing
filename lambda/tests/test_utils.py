"""Unit tests for helper utilities in lambda/utils.py."""

from __future__ import annotations

import re
import sys
import types
import unittest
from unittest import mock


# Stub awswrangler before importing the module under test.
awswrangler_module = sys.modules.get("awswrangler")
if awswrangler_module is None:
    awswrangler_module = types.ModuleType("awswrangler")
    sys.modules["awswrangler"] = awswrangler_module

if not hasattr(awswrangler_module, "s3"):
    awswrangler_module.s3 = types.SimpleNamespace()

if not hasattr(awswrangler_module.s3, "read_csv"):
    awswrangler_module.s3.read_csv = lambda *args, **kwargs: []  # type: ignore[attr-defined]

if not hasattr(awswrangler_module.s3, "read_parquet"):
    awswrangler_module.s3.read_parquet = lambda *args, **kwargs: []  # type: ignore[attr-defined]


from utils import create_job_name, load_files_in_chunks, split_s3_uri  # noqa: E402  pylint: disable=wrong-import-position


class CreateJobNameTests(unittest.TestCase):
    """Tests covering job name generation logic."""

    def test_generates_valid_length_and_characters(self) -> None:
        prefix = "My Job Prefix With Spaces & Symbols!"
        name = create_job_name(prefix, index=42)

        self.assertLessEqual(len(name), 128)
        self.assertRegex(name, r"^[A-Za-z0-9\-]+$")
        self.assertRegex(name, r"-part-0042-[A-Za-z0-9]{6}$")


class SplitS3UriTests(unittest.TestCase):
    """Exercise S3 URI parsing utility."""

    def test_splits_valid_uri(self) -> None:
        bucket, key = split_s3_uri("s3://bucket/path/to/file")
        self.assertEqual(bucket, "bucket")
        self.assertEqual(key, "path/to/file")

    def test_raises_for_invalid_uri(self) -> None:
        with self.assertRaises(ValueError):
            split_s3_uri("http://example.com/object")

        with self.assertRaises(ValueError):
            split_s3_uri("s3://bucket-only")


class LoadFilesInChunksTests(unittest.TestCase):
    """Verify chunked loaders delegate to awswrangler appropriately."""

    def setUp(self) -> None:
        self.read_csv_patch = mock.patch("utils.wr.s3.read_csv")
        self.read_parquet_patch = mock.patch("utils.wr.s3.read_parquet")

    def tearDown(self) -> None:
        mock.patch.stopall()

    def test_loads_csv_chunks(self) -> None:
        mock_read_csv = self.read_csv_patch.start()
        mock_read_csv.return_value = [["chunk1"], ["chunk2"]]

        results = list(
            load_files_in_chunks("s3://bucket/input.csv", "csv", chunk_size=100)
        )

        self.assertEqual(results, list(enumerate(mock_read_csv.return_value)))
        mock_read_csv.assert_called_once_with("s3://bucket/input.csv", chunksize=100)

    def test_loads_parquet_chunks(self) -> None:
        mock_read_parquet = self.read_parquet_patch.start()
        mock_read_parquet.return_value = [["pchunk"]]

        results = list(
            load_files_in_chunks("s3://bucket/input.parquet", "parquet", chunk_size=256)
        )

        self.assertEqual(results, list(enumerate(mock_read_parquet.return_value)))
        mock_read_parquet.assert_called_once_with(
            "s3://bucket/input.parquet", chunked=256
        )

    def test_unsupported_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            list(load_files_in_chunks("s3://bucket/input.txt", "txt", chunk_size=10))


if __name__ == "__main__":
    unittest.main()
