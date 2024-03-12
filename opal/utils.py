import logging

from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


def db_conn():
    logger.info(f"Connecting to root@localhost:3308/osu")
    return create_engine("mysql+pymysql://root@localhost:3308/osu")
