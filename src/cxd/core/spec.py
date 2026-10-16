from pathlib import Path
from typing import Any

import pandas as pd
from qorageom.spec import Spec, SpecScan

__all__ = ["Spec", "SpecScan", "get_positions"]


def get_positions(run_ns: list[int], spec_file: Path) -> pd.DataFrame:
    data = []
    with Spec(spec_file) as sp:
        for run_n in run_ns:
            scan = sp.get_scan(run_n)
            row: dict[str, Any] = {"run_n": run_n, **scan.motors}
            row['sdd'] = scan.sdd
            row['bc'] = scan.beam_center
            data.append(row)
    df: pd.DataFrame = pd.DataFrame(data).set_index('run_n')
    return df
