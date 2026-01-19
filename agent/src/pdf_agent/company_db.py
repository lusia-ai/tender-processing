from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import os

import psycopg
from psycopg.errors import UndefinedTable
from psycopg.rows import dict_row

import logging

_LOGGER = logging.getLogger("pdf_agent.company_db")


def _db_dsn() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    host = os.getenv("DB_HOST", "db")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "tender_company")
    user = os.getenv("DB_USER", "tender")
    password = os.getenv("DB_PASSWORD", "tender")
    return f"host={host} port={port} dbname={name} user={user} password={password}"


@contextmanager
def _connect():
    conn = psycopg.connect(_db_dsn(), row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


def get_company_profile() -> Dict[str, Any]:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, description, country, website, notes, source_notes "
            "FROM company_profile ORDER BY id ASC LIMIT 1"
        )
        profile = cur.fetchone()
        if not profile:
            return {}
        cur.execute(
            "SELECT capability, capacity, certification, source_notes "
            "FROM company_capability WHERE profile_id = %s ORDER BY id ASC",
            (profile["id"],),
        )
        capabilities = cur.fetchall()
    profile["capabilities"] = capabilities
    return profile


def list_tenders(limit: int = 20) -> List[Dict[str, Any]]:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT tender_id, source_file, primary_title, summary, notes "
            "FROM tenders ORDER BY tender_id ASC LIMIT %s",
            (limit,),
        )
        return cur.fetchall()


def get_tender_lots(tender_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT tender_id, lot_number, title, description, estimated_value, "
            "estimated_currency, qualification_min_value, qualification_currency, pages, source_file "
            "FROM tender_lots WHERE tender_id = %s ORDER BY lot_number ASC LIMIT %s",
            (tender_id, limit),
        )
        return cur.fetchall()


def search_lots(keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT tender_id, lot_number, title, description, pages, source_file "
            "FROM tender_lots "
            "WHERE title ILIKE %s OR description ILIKE %s "
            "ORDER BY tender_id ASC LIMIT %s",
            (f"%{keyword}%", f"%{keyword}%", limit),
        )
        return cur.fetchall()


def get_tender_activity(tender_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    with _connect() as conn:
        cur = conn.cursor()
        if tender_id:
            cur.execute(
                "SELECT tender_id, status, notes, source_notes "
                "FROM tender_activity WHERE tender_id = %s ORDER BY id ASC LIMIT %s",
                (tender_id, limit),
            )
        else:
            cur.execute(
                "SELECT tender_id, status, notes, source_notes "
                "FROM tender_activity ORDER BY tender_id ASC LIMIT %s",
                (limit,),
            )
        return cur.fetchall()


def get_company_deliveries(
    tender_id: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    clauses = []
    params: List[Any] = []
    if tender_id:
        clauses.append("tender_id = %s")
        params.append(tender_id)
    if keyword:
        clauses.append("(title ILIKE %s OR scope ILIKE %s OR customer ILIKE %s)")
        token = f"%{keyword}%"
        params.extend([token, token, token])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT tender_id, title, customer, delivered_at, value, currency, scope, evidence, source_notes "
                f"FROM company_delivery {where} "
                "ORDER BY delivered_at DESC NULLS LAST LIMIT %s",
                (*params, limit),
            )
            return cur.fetchall()
        except UndefinedTable:
            _LOGGER.warning("company_delivery table missing; returning empty delivery history")
            return []
