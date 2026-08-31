"""Streamlit read layer over the HTTP API.

Two views. The dashboard carries the headline figures, the Locale, Intent and Provider mode
breakdowns, and the Diagnosis with the Evidence behind each Finding; the drill-down shows every
Trial for one Query with raw Answer text, highlighted Mentions and clickable Citations. The
drill-down is the one that matters: it is how a reader confirms the numbers describe real text.

The UI never calls a Provider directly. It reads the API, whose only live path is the ad-hoc Query.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
import streamlit as st


DEFAULT_API = "http://127.0.0.1:8000"
HIGHLIGHT = "background-color: #ffd54f; color: #000; padding: 0 2px; border-radius: 2px"


def highlight_mentions(text: str, aliases: list[str]) -> str:
    """Wrap each Alias occurrence so a reader can see exactly what was matched."""
    escaped = st_escape(text)
    for alias in sorted(aliases, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w)({re.escape(st_escape(alias))})(?!\w)", re.IGNORECASE)
        escaped = pattern.sub(rf'<mark style="{HIGHLIGHT}">\1</mark>', escaped)
    return escaped


def st_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fetch(api: str, path: str) -> Any:
    response = httpx.get(f"{api}{path}", timeout=30.0)
    response.raise_for_status()
    return response.json()


def main() -> None:  # pragma: no cover - exercised by running the app
    st.set_page_config(page_title="Boutiqaat AI Search Visibility", layout="wide")
    api = st.sidebar.text_input("API base URL", DEFAULT_API)
    st.sidebar.caption(
        "Runs are started from the CLI, not here: a Run is minutes of paid calls (ADR-0003)."
    )
    try:
        runs = fetch(api, "/runs")
    except Exception as error:  # noqa: BLE001 - surfaced to the reader, not swallowed
        st.error(f"Cannot reach the API at {api}: {error}")
        st.stop()
        return
    if not runs:
        st.warning("No Runs are stored yet. Record one with `python -m avi.cli run`.")
        return

    run_id = st.sidebar.selectbox("Run", [run["id"] for run in runs])
    view = st.sidebar.radio("View", ["Dashboard", "Query drill-down"])
    metrics = fetch(api, f"/runs/{run_id}/metrics")

    if view == "Dashboard":
        render_dashboard(
            metrics,
            fetch(api, f"/runs/{run_id}/slices"),
            fetch(api, f"/runs/{run_id}/diagnosis"),
        )
    else:
        render_drilldown(api, run_id, metrics)


def render_dashboard(
    metrics: dict[str, Any],
    slices: dict[str, Any],
    findings: list[dict[str, Any]],
) -> None:  # pragma: no cover
    st.title("Boutiqaat AI Search Visibility")
    st.caption("Findings describe OpenAI's models, not AI search in general.")

    rate = metrics["visibility_rate"]
    share = metrics["share_of_voice"]
    left, middle, right = st.columns(3)
    left.metric(
        "Visibility Rate",
        f"{(rate['value'] or 0):.1%}",
        f"{rate['mentioned']} of {rate['relevant_trials']} relevant Trials",
    )
    middle.metric(
        "Share of Voice",
        f"{(share['value'] or 0):.1%}",
        f"{share['boutiqaat_mentions']} of {share['seed_mentions']} Brand Mentions",
    )
    buckets = {"always": 0, "sometimes": 0, "never": 0}
    for item in metrics["consistency"]:
        buckets[item["bucket"]] += 1
    right.metric(
        "Consistency",
        f"{buckets['sometimes']} sometimes",
        f"{buckets['always']} always, {buckets['never']} never",
    )

    st.subheader("Visibility by Locale, Intent and Provider mode")
    st.caption(
        "Trials on irrelevant Queries are excluded: absence from a question Boutiqaat could not "
        "answer is not a failure."
    )
    for keyword, heading in (
        ("locale", "Locale"),
        ("intent", "Intent"),
        ("provider_mode", "Provider mode"),
    ):
        st.markdown(f"**{heading}**")
        st.dataframe(
            [
                {
                    heading: row["value"],
                    "Mentioned": row["mentioned"],
                    "Relevant Trials": row["relevant_trials"],
                    "Visibility Rate": f"{(row['visibility_rate'] or 0):.1%}",
                }
                for row in slices[keyword]
            ],
            use_container_width=True,
        )

    st.subheader("Diagnosis")
    if not findings:
        st.info("No candidate cause was supported by Evidence observed in this Run.")
    else:
        st.caption(
            "Each claim rests on Evidence recorded in this Run. A cause with no supporting "
            "Evidence is not shown at all."
        )
        for finding in findings:
            with st.expander(finding["cause"], expanded=False):
                st.write(finding["statement"])
                if finding["fetched_page_count"] or finding["unfetched_page_count"]:
                    st.caption(
                        f"{finding['fetched_page_count']} cited pages fetched, "
                        f"{finding['unfetched_page_count']} unfetched and excluded from this claim."
                    )
                if finding["answer_ids"]:
                    st.caption(
                        "Evidence, Answer ids: "
                        + ", ".join(str(i) for i in finding["answer_ids"])
                    )
                for url in finding["citation_urls"][:10]:
                    st.markdown(f"- [{url}]({url})")
                st.markdown(f"**What would have to change:** {finding['remedy']}")

    st.subheader("Consistency by Query")
    st.caption(
        "A Query that is sometimes visible is a different commercial problem from one that never is."
    )
    st.dataframe(
        [
            {
                "Query": item["query_id"],
                "Mode": item["provider_mode"],
                "Consistency": item["bucket"],
                "Trials": len(item["answer_ids"]),
            }
            for item in metrics["consistency"]
        ],
        use_container_width=True,
    )

    st.subheader("Recommendation Strength")
    st.caption("A distribution, never an average: the labels are ordinal.")
    st.dataframe(
        [
            {"Strength": strength, "Answers": count}
            for strength, count in metrics["recommendation_strength"].items()
        ],
        use_container_width=True,
    )


def render_drilldown(api: str, run_id: str, metrics: dict[str, Any]) -> None:  # pragma: no cover
    st.title("Query drill-down")
    query_ids = sorted({item["query_id"] for item in metrics["consistency"]})
    if not query_ids:
        st.warning("This Run has no Relevant Query Trials.")
        return
    query_id = st.selectbox("Query", query_ids)
    payload = fetch(api, f"/runs/{run_id}/queries/{query_id}")
    aliases = ["Boutiqaat", "Boutiqat", "Boutiquaat", "بوتيكات", "boutiqaat.com", "boutiqat.com"]

    for trial in payload["trials"]:
        header = (
            f"{trial['provider_mode']} · Trial {trial['trial_index']} · "
            f"{'Mentioned' if trial['mentioned'] else 'Absent'}"
        )
        with st.expander(header, expanded=trial["mentioned"]):
            st.caption(
                f"Answer id {trial['answer_id']} · {trial['model_identifier']} · "
                f"search performed: {'yes' if trial['search_performed'] else 'no'} · "
                f"Recommendation Strength: {trial['recommendation_strength'] or 'not judged'}"
            )
            st.markdown(highlight_mentions(trial["text"], aliases), unsafe_allow_html=True)
            if trial["citations"]:
                st.markdown("**Citations**")
                for citation in trial["citations"]:
                    status = citation["page_status"] or "not fetched"
                    st.markdown(
                        f"- [{st_escape(citation['title'])}]({citation['url']}) "
                        f"· {citation['source_type']} · page: {status}"
                    )


if __name__ == "__main__":  # pragma: no cover
    main()
