from prefect import task
from sqlalchemy import create_engine


@task(name="Create DB Connection")
def db_conn():
    return create_engine("mysql+pymysql://root@localhost:3308/osu")
