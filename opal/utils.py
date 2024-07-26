import logging
import sqlite3
from functools import cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
PROJ_DIR = Path(__file__).parent.parent
RSC_DIR = PROJ_DIR / "rsc"


def db_conn():
    logger.info(f"Connecting to root@localhost:3308/osu")
    return sqlite3.connect(RSC_DIR / "2024_05_10_top_1000_mania.db")


@cache
def read_database(key: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        f"SELECT * FROM osu WHERE `keys` = {key}", db_conn()
    )
    return df.astype({"keys": int})
