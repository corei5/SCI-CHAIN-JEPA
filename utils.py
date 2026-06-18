"""utils.py -- shared helpers (year parsing, field reading)."""
import re
from typing import Optional, Dict
from config import CFG


def parse_year(row) -> Optional[int]:
    for col in (CFG.year_column, "source_year"):
        val = row.get(col)
        if val is None:
            continue
        try:
            if isinstance(val, str):
                m = re.search(r"\d{4}", val)
                if m:
                    return int(m.group())
            else:
                return int(float(val))
        except (ValueError, TypeError):
            continue
    return None


def get_field(chain: Dict) -> str:
    f = chain.get("field")
    return f if f and len(str(f).strip()) > 0 else "unknown"
