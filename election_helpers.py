# Name: Sumnima Wuni | Index Number: 10022300194
"""Inject aggregated national totals from Ghana_Election_Result.csv when relevant."""
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


_ELECTION_CSV = "Ghana_Election_Result.csv"


def _election_intent(query: str) -> bool:
    q = query.lower()
    keys = (
        "vote",
        "votes",
        "election",
        "candidate",
        "president",
        "won",
        "win",
        "winner",
        "highest",
        "most",
        "ndc",
        "npp",
    )
    return any(k in q for k in keys)


def _year_from_query(query: str) -> Optional[int]:
    m = re.search(r"\b(19\d{2}|20\d{2})\b", query)
    if m:
        return int(m.group(1))
    return None


def build_election_aggregate_chunk(docs_path: Path, year: int) -> Optional[Dict[str, object]]:
    csv_path = docs_path / _ELECTION_CSV
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "Year" not in df.columns or "Votes" not in df.columns:
        return None
    df = df[df["Year"] == year]
    if df.empty:
        return None
    df = df.copy()
    df["Votes_num"] = pd.to_numeric(
        df["Votes"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    ).fillna(0)
    totals = (
        df.groupby(["Candidate", "Party"], as_index=False)["Votes_num"]
        .sum()
        .sort_values("Votes_num", ascending=False)
        .reset_index(drop=True)
    )
    if totals.empty:
        return None
    top = totals.iloc[0]
    second = totals.iloc[1] if len(totals) > 1 else None
    lines = [
        f"AGGREGATED NATIONAL TOTALS (computed by summing regional Votes in {_ELECTION_CSV} for Year={year}).",
        f"Highest total votes: {top['Candidate']} ({top['Party']}) — {int(top['Votes_num']):,} votes.",
    ]
    if second is not None:
        lines.append(
            f"Second: {second['Candidate']} ({second['Party']}) — {int(second['Votes_num']):,} votes."
        )
    lines.append(
        "Use this block for questions about who won or who had the most votes nationally for that year."
    )
    text = "\n".join(lines)
    return {
        "chunk_id": f"csv-aggregate-{year}",
        "source": _ELECTION_CSV,
        "text": text,
        "combined_score": 1.0,
        "vector_score": 1.0,
        "keyword_score": 1.0,
        "metadata": {"doc_type": "csv_aggregate", "page": "computed", "year": str(year)},
    }


def _default_election_year(docs_path: Path) -> Optional[int]:
    csv_path = docs_path / _ELECTION_CSV
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path, usecols=["Year"])
        return int(df["Year"].max())
    except Exception:
        return None


def inject_election_aggregate(
    query: str, retrieved: List[Dict[str, object]], docs_path: str
) -> List[Dict[str, object]]:
    if not _election_intent(query):
        return retrieved
    root = Path(docs_path)
    year = _year_from_query(query)
    if year is None:
        year = _default_election_year(root)
    if year is None:
        return retrieved
    agg = build_election_aggregate_chunk(root, year)
    if agg is None:
        return retrieved
    rest = [r for r in retrieved if r.get("chunk_id") != agg["chunk_id"]]
    return [agg] + rest
