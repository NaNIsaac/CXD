from pathlib import Path

from cxd.core.spec import get_positions


def test_get_positions_returns_indexed_dataframe(spec_file: Path):
    df = get_positions([1, 2], spec_file)
    assert df.index.name == "run_n"
    assert list(df.index) == [1, 2]
    assert df.loc[1, "mot_a"] == 0.1
    assert df.loc[2, "mot_b"] == 1.2
    assert df.loc[1, "sdd"] == 123.45
    assert df.loc[1, "bc"] == {"x": 512, "y": 256}
