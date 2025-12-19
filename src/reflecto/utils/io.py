from pathlib import Path

import pandas as pd

def read_dat(file: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        file, 
        header=None,
        names=['tth', 'intensity'],
        sep='\\s+'
    )
    return df