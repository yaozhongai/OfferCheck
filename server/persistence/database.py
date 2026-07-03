"""
SQLAlchemy Engine + Session 管理
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from server.config import get_config

_engine = None
_SessionLocal = None
Base = declarative_base()


def get_engine():
    global _engine
    if _engine is None:
        config = get_config()
        db_path = config.db_path
        _engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )

        @event.listens_for(_engine, "connect")
        def _set_wal(dbapi_conn, _connection_record):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return _engine


def get_session():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal()


def init_db():
    """创建所有表"""
    from server.persistence.models import Base  # noqa
    Base.metadata.create_all(bind=get_engine())


def reset_engine():
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None
