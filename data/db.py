"""
Phase 5 - Database Layer
Persists every round to SQLite via SQLAlchemy.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import DATABASE_URL

logger = logging.getLogger(__name__)


def _normalize_source(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return normalized or None


class Base(DeclarativeBase):
    pass


class RoundRecord(Base):
    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String, nullable=False)
    multiplier = Column(Float, nullable=False)
    color = Column(String, nullable=False)
    source_file = Column(String, nullable=False)
    source = Column(String, nullable=False, default="aviator")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("timestamp", "multiplier", name="uq_timestamp_multiplier"),
        Index("ix_source_timestamp", "source", "timestamp"),
        Index("ix_source", "source"),
        Index("ix_source_id", "source", "id"),
    )

    def __repr__(self):
        return f"<Round id={self.id} ts={self.timestamp} mult={self.multiplier}>"


class ReplayResultRecord(Base):
    __tablename__ = "replay_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feature = Column(String, nullable=False)
    sessions_tested = Column(Integer, nullable=False, default=0)
    scored_predictions = Column(Integer, nullable=False, default=0)
    accuracy = Column(Float, nullable=False, default=0.0)
    status = Column(String, nullable=False, default="ok")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<ReplayResult id={self.id} feature={self.feature} accuracy={self.accuracy}>"


# Engine and session factory
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False, "timeout": 30},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        # Enable WAL mode for concurrent read+write access from multiple threads/processes
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA busy_timeout=30000"))
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(rounds)"))}
        if "source" not in columns:
            conn.execute(text("ALTER TABLE rounds ADD COLUMN source VARCHAR NOT NULL DEFAULT 'aviator'"))
        index_names = {row[1] for row in conn.execute(text("PRAGMA index_list(rounds)"))}
        if "ix_source_id" not in index_names:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_source_id ON rounds(source, id)"))
    logger.info("Database initialized")


# Create tables on import so first-use persistence works without a separate init call.
init_db()


def insert_rounds(rounds) -> dict:
    """
    Insert a list of Round objects into the database.
    Skips duplicates silently.
    Returns a dict with inserted and duplicate counts.
    """
    inserted = 0
    duplicates = 0

    with SessionLocal() as session:
        for r in rounds:
            record = RoundRecord(
                timestamp=r.timestamp,
                multiplier=r.multiplier,
                color=r.color,
                source_file=r.source_file,
                source=_normalize_source(r.source) or "aviator",
            )
            session.add(record)
            try:
                session.flush()
                inserted += 1
            except IntegrityError:
                session.rollback()
                duplicates += 1

        session.commit()

    logger.info(f"Inserted {inserted} rows, skipped {duplicates} duplicates")
    return {"inserted": inserted, "duplicates": duplicates}


def get_all_rounds(source: Optional[str] = None) -> List[dict]:
    """Return all rounds as a list of dicts, optionally filtered by source."""
    with SessionLocal() as session:
        query = session.query(RoundRecord)
        source = _normalize_source(source)
        if source:
            query = query.filter(RoundRecord.source == source)
        records = query.order_by(RoundRecord.id).all()
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "multiplier": r.multiplier,
                "color": r.color,
                "source_file": r.source_file,
                "source": r.source,
                "created_at": str(r.created_at),
            }
            for r in records
        ]


def get_latest_rounds(n: int = 100, source: Optional[str] = None) -> List[dict]:
    """Return the N most recent rounds, optionally filtered by source."""
    with SessionLocal() as session:
        query = session.query(RoundRecord)
        source = _normalize_source(source)
        if source:
            query = query.filter(RoundRecord.source == source)
        records = query.order_by(RoundRecord.id.desc()).limit(n).all()
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "multiplier": r.multiplier,
                "color": r.color,
                "source_file": r.source_file,
                "source": r.source,
                "created_at": str(r.created_at),
            }
            for r in reversed(records)
        ]


def get_recent_rounds(n: int = 500, source: Optional[str] = None) -> List[dict]:
    """Return a recent window of rounds using the latest IDs, avoiding full-table scans."""
    if n <= 0:
        return []
    with SessionLocal() as session:
        query = session.query(RoundRecord)
        source = _normalize_source(source)
        if source:
            query = query.filter(RoundRecord.source == source)
        records = query.order_by(RoundRecord.id.desc()).limit(n).all()
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "multiplier": r.multiplier,
                "color": r.color,
                "source_file": r.source_file,
                "source": r.source,
                "created_at": str(r.created_at),
            }
            for r in reversed(records)
        ]


def save_replay_result(payload: dict) -> dict:
    """Persist a replay evaluation payload and return the stored record."""
    init_db()
    with SessionLocal() as session:
        record = ReplayResultRecord(
            feature=payload.get("feature", "unknown"),
            sessions_tested=int(payload.get("sessions_tested", 0)),
            scored_predictions=int(payload.get("scored_predictions", 0)),
            accuracy=float(payload.get("accuracy", 0.0)),
            status=payload.get("status", "ok"),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return {
            "id": record.id,
            "feature": record.feature,
            "sessions_tested": record.sessions_tested,
            "scored_predictions": record.scored_predictions,
            "accuracy": record.accuracy,
            "status": record.status,
            "created_at": str(record.created_at),
        }


def get_replay_results(limit: int = 50) -> List[dict]:
    """Return recent replay result rows in ascending id order."""
    init_db()
    with SessionLocal() as session:
        records = (
            session.query(ReplayResultRecord)
            .order_by(ReplayResultRecord.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "feature": r.feature,
                "sessions_tested": r.sessions_tested,
                "scored_predictions": r.scored_predictions,
                "accuracy": r.accuracy,
                "status": r.status,
                "created_at": str(r.created_at),
            }
            for r in reversed(records)
        ]
