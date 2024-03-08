import numpy as np
import pandas as pd

from opal.utils import db_conn

MAX = 320


def read_maps(conn):
    df = pd.read_sql(
        r"SELECT * FROM osu_beatmaps WHERE playmode = 3 "
        r"AND difficultyrating > 2.5 ",
        conn,
    ).assign(
        ln_ratio=lambda x: x.countSlider / (x.countSlider + x.countNormal),
        mapname=lambda x: x.filename.str[:-4],
    )[
        [
            "beatmap_id",
            "ln_ratio",
            "mapname",
            "diff_size",
            "diff_overall",
            "approved",
            "difficultyrating",
            "playcount",
            "passcount",
        ]
    ]
    return df


def read_plays(conn):
    return pd.read_sql(
        r"SELECT * FROM osu_scores_mania_high WHERE score > 600000",
        conn,
    ).assign(
        accuracy=lambda x: (
            x.countgeki * 1
            + x.count300 * 300 / MAX
            + x.countkatu * 200 / MAX
            + x.count100 * 100 / MAX
            + x.count50 * 50 / MAX
        )
        / (
            x.countgeki
            + x.count300
            + x.countkatu
            + x.count100
            + x.count50
            + x.countmiss
        ),
        speed=lambda x: (
            np.where(
                (x.enabled_mods & (1 << 6)) > 0,
                1,
                np.where((x.enabled_mods & (1 << 8)) > 0, -1, 0),
            )
        ),
        year=lambda x: x.date.dt.year,
    )[
        [
            "score_id",
            "beatmap_id",
            "user_id",
            "score",
            "accuracy",
            "speed",
            "year",
            "pp",
        ]
    ]


def read_players(conn):
    return pd.read_sql(
        r"SELECT * FROM sample_users",
        conn,
    )[
        [
            "user_id",
            "username",
        ]
    ]


def merge(df_maps, df_plays, df_users):
    return (
        (
            df_plays.merge(df_maps, on="beatmap_id").merge(
                df_users, on="user_id"
            )
        )
        .rename(
            {
                "score_id": "sid",
                "beatmap_id": "mid",
                "user_id": "uid",
                "diff_size": "keys",
                "diff_overall": "od",
                "difficultyrating": "sr",
            },
            axis=1,
        )
        .set_index("sid")
    )


def write_dataset(df, conn):
    df.to_sql(
        "osu_dataset",
        conn,
        schema="osu",
        if_exists="replace",
    )


def create_dataset():
    conn = db_conn()
    df_maps = read_maps(conn)
    df_plays = read_plays(conn)
    df_users = read_players(conn)
    df = merge(df_maps, df_plays, df_users)
    df = (
        df.merge(
            df["mid"].value_counts(normalize=True).rename("p_mid"),
            left_on="mid",
            right_index=True,
        )
        .merge(
            df["uid"].value_counts(normalize=True).rename("p_uid"),
            left_on="uid",
            right_index=True,
        )
        .assign(
            prob=lambda x: x["p_mid"] * x["p_uid"],
            rand=lambda x: np.random.uniform(0, 1, len(x)),
        )
    )
    write_dataset(df, conn)
    conn.dispose()


if __name__ == "__main__":
    create_dataset()
