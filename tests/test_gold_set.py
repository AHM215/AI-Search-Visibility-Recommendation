"""Judge agreement against the Gold Set.

Opt-in: it reads the recorded corpus database, which is not part of the committed fixture corpus,
and it measures a model's labelling rather than the pipeline's behaviour. Run with:

    pytest -m gold_set --gold-database corpus2.db
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
AGREEMENT_FLOOR = 0.80

pytestmark = pytest.mark.gold_set


def load_labels() -> dict[int, str]:
    document = yaml.safe_load((ROOT / "gold_set.yaml").read_text(encoding="utf-8"))
    return {entry["answer_id"]: entry["label"] for entry in document["labels"]}


def test_the_gold_set_is_stratified_and_sized() -> None:
    labels = load_labels()

    assert len(labels) >= 20
    assert len(set(labels.values())) >= 3


def test_judge_agreement_stays_above_the_floor(gold_database: Path) -> None:
    labels = load_labels()
    connection = sqlite3.connect(gold_database)
    judged = dict(
        connection.execute("SELECT answer_id, recommendation_strength FROM verdicts").fetchall()
    )
    connection.close()

    comparable = {
        answer_id: label
        for answer_id, label in labels.items()
        if label != "not_a_mention" and answer_id in judged
    }
    assert comparable, "no comparable labels; is the gold database the recorded corpus?"

    matrix: Counter[tuple[str, str]] = Counter()
    for answer_id, human in comparable.items():
        matrix[(human, judged[answer_id])] += 1
    agreed = sum(count for (human, model), count in matrix.items() if human == model)
    agreement = agreed / len(comparable)

    print("\nconfusion matrix (human -> judge):")
    for (human, model), count in sorted(matrix.items()):
        marker = "==" if human == model else "!="
        print(f"  {human:<12} {marker} {model:<12} {count}")
    print(f"agreement: {agreed}/{len(comparable)} = {agreement:.1%}")

    assert agreement >= AGREEMENT_FLOOR


def test_alias_false_positives_are_recorded_not_hidden() -> None:
    labels = load_labels()

    false_positives = [
        answer_id for answer_id, label in labels.items() if label == "not_a_mention"
    ]

    assert false_positives, (
        "The Gold Set must record Alias false positives where they exist: they inflate "
        "Visibility Rate and are invisible to any metric computed from Mentions alone."
    )
