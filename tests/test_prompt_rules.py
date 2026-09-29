from __future__ import annotations

import unittest

from elt_taskgen.destinations import Destination
from elt_taskgen.training.dbt_runner import rewrite_model_sql

from elt_environment.prompt import SYSTEM_PROMPT

SOURCE = "{{ source('raw', 't') }}"

ALLOWED = {
    "QUALIFY": f"SELECT a FROM {SOURCE} QUALIFY ROW_NUMBER() OVER (PARTITION BY b ORDER BY c) = 1",
    "COUNT(DISTINCT": f"SELECT COUNT(DISTINCT a) AS x FROM {SOURCE}",
    "FILTER (WHERE": f"SELECT SUM(a) FILTER (WHERE b > 0) AS x FROM {SOURCE}",
    "TRY_CAST": f"SELECT TRY_CAST(a AS INT) AS x FROM {SOURCE}",
    "DATE_TRUNC": f"SELECT DATE_TRUNC('day', ts) AS x FROM {SOURCE}",
    "DATEDIFF": f"SELECT DATEDIFF(day, a, b) AS x FROM {SOURCE}",
    "LAST_VALUE": (
        f"SELECT LAST_VALUE(a) OVER (ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING"
        f" AND UNBOUNDED FOLLOWING) AS x FROM {SOURCE}"
    ),
    "REGEXP_REPLACE": f"SELECT REGEXP_REPLACE(a, 'x', 'y') AS x FROM {SOURCE}",
    "TO_CHAR": f"SELECT TO_CHAR(ts, 'YYYY-MM') AS x FROM {SOURCE}",
    "MD5 over CAST(... AS VARCHAR)": f"SELECT MD5(CAST(a AS VARCHAR)) AS x FROM {SOURCE}",
}

REFUSED = {
    "UPPER": f"SELECT UPPER(a) AS x FROM {SOURCE}",
    "ABS": f"SELECT ABS(a) AS x FROM {SOURCE}",
    "GREATEST": f"SELECT GREATEST(a, b) AS x FROM {SOURCE}",
    "LEAST": f"SELECT LEAST(a, b) AS x FROM {SOURCE}",
    "EXISTS": f"SELECT a FROM {SOURCE} AS p WHERE EXISTS (SELECT 1 FROM {SOURCE} AS q WHERE q.a = p.a)",
    "BOOL_AND": f"SELECT BOOL_AND(a) AS x FROM {SOURCE}",
    "TO_NUMBER": f"SELECT TO_NUMBER(a) AS x FROM {SOURCE}",
    "CONCAT_WS": f"SELECT CONCAT_WS('-', a, b) AS x FROM {SOURCE}",
    "SPLIT_PART": f"SELECT SPLIT_PART(a, '-', 1) AS x FROM {SOURCE}",
}


def _accepted(sql: str) -> bool:
    try:
        rewrite_model_sql(sql, Destination.SNOWFLAKE, json_columns=frozenset())
    except Exception:
        return False
    return True


class PromptSqlRuleTests(unittest.TestCase):
    def test_constructs_the_prompt_allows_are_accepted_by_the_grader(self) -> None:
        for name, sql in ALLOWED.items():
            with self.subTest(construct=name):
                self.assertIn(name, SYSTEM_PROMPT)
                self.assertTrue(_accepted(sql))

    def test_constructs_the_prompt_names_as_refused_are_refused_by_the_grader(self) -> None:
        for name, sql in REFUSED.items():
            with self.subTest(construct=name):
                self.assertIn(name, SYSTEM_PROMPT)
                self.assertFalse(_accepted(sql))


if __name__ == "__main__":
    unittest.main()
