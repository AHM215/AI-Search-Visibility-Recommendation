from __future__ import annotations

from avi.ui import highlight_mentions, st_escape


ALIASES = ["Boutiqaat", "Boutiqat", "بوتيكات", "boutiqat.com"]


def test_a_mention_is_wrapped_so_a_reader_can_see_what_matched() -> None:
    rendered = highlight_mentions("Try Boutiqaat for beauty.", ALIASES)

    assert "<mark" in rendered
    assert "Boutiqaat</mark>" in rendered


def test_arabic_aliases_are_highlighted_too() -> None:
    rendered = highlight_mentions("بوتيكات متجر تجميل", ALIASES)

    assert "<mark" in rendered


def test_text_without_a_mention_is_left_unmarked() -> None:
    rendered = highlight_mentions("Sephora and iHerb are options.", ALIASES)

    assert "<mark" not in rendered


def test_answer_text_is_escaped_before_it_is_marked_up() -> None:
    rendered = highlight_mentions("<script>alert(1)</script> Boutiqaat", ALIASES)

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_escaping_handles_ampersands() -> None:
    assert st_escape("Faces & Co") == "Faces &amp; Co"
