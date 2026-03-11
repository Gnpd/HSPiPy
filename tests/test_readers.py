"""Tests for HSP file readers."""
import textwrap
import pytest
import pandas as pd

from hspipy.readers import CSVReader, HSPDataReader


CSV_PATH = "examples/hsp_example.csv"

REQUIRED_COLUMNS = {"Solvent", "D", "P", "H", "Score"}


def test_csv_reader_columns():
    reader = CSVReader()
    df = reader.read(CSV_PATH)
    assert REQUIRED_COLUMNS.issubset(set(df.columns))


def test_csv_reader_dtypes():
    reader = CSVReader()
    df = reader.read(CSV_PATH)
    for col in ("D", "P", "H", "Score"):
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"


def test_csv_reader_no_empty_rows():
    reader = CSVReader()
    df = reader.read(CSV_PATH)
    assert df[["D", "P", "H"]].isna().sum().sum() == 0, "D/P/H must have no NaN after loading"


def test_hspdatareader_loads_csv():
    reader = HSPDataReader()
    df = reader.read(CSV_PATH)
    assert len(df) > 0
    assert REQUIRED_COLUMNS.issubset(set(df.columns))


def test_hspdatareader_file_not_found():
    reader = HSPDataReader()
    with pytest.raises(FileNotFoundError):
        reader.read("does_not_exist.csv")


def test_csv_reader_empty_file(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    reader = CSVReader()
    with pytest.raises(ValueError, match="empty"):
        reader.read(empty)


def test_csv_reader_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Solvent,D,P\nwater,15.5,16.0\n")
    reader = CSVReader()
    with pytest.raises(ValueError):
        reader.read(bad)


def test_csv_reader_with_headerless_file(tmp_path):
    """A CSV without headers should still be parsed if it has 5+ columns."""
    content = textwrap.dedent("""\
        Water,15.5,16.0,42.3,1
        Ethanol,15.8,8.8,19.4,1
        Hexane,14.9,0.0,0.0,0
    """)
    f = tmp_path / "nohdr.csv"
    f.write_text(content)
    reader = CSVReader()
    df = reader.read(f)
    assert len(df) == 3
    assert REQUIRED_COLUMNS.issubset(set(df.columns))
