"""
test_predictions.py — Regression suite for the fake-news detector.

Uses paragraph-length test cases (not bare headlines) since the model is
trained on title x2 + full article body, and bare headlines produce
out-of-distribution TF-IDF vectors (see Session Summary, Step 3 finding).

Two modes:
  1. pytest test_predictions.py          -> soft assertions, CI-friendly
  2. python test_predictions.py          -> prints a markdown results table
     you can paste straight into the README's "known limitations" section

Soft assertions check *direction* (label) not exact confidence, since RF
output will drift a few points across retrains even with the same data.
"""

import os
import pickle
import sys

import pytest

# ── Locate model artifacts ───────────────────────────────────────────
# Adjust ROOT if your .pkl files live somewhere else.
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_DIRS = [
    os.path.join(HERE, "backend"),
    HERE,
    os.path.join(HERE, "..", "backend"),
]

sys.path.insert(0, HERE)
try:
    from backend.src.preprocess import clean_text
except ImportError:
    # Fallback: allow running from inside backend/
    sys.path.insert(0, os.path.join(HERE, "backend"))
    from src.preprocess import clean_text  # type: ignore


def _find_file(filename):
    for d in CANDIDATE_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename} in any of {CANDIDATE_DIRS}. "
        f"Run main.py first, or edit CANDIDATE_DIRS in this script."
    )


def load_artifacts(model_name="random_forest"):
    vec_path = _find_file("vectorizer.pkl")

    # Project now consolidates to a single model.pkl (Random Forest).
    # Try that first; fall back to a per-model file if one exists.
    try:
        model_path = _find_file("model.pkl")
    except FileNotFoundError:
        model_path = _find_file(f"model_{model_name}.pkl")

    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model, vectorizer


def predict(text, model, vectorizer, title=""):
    combined = title + " " + title + " " + text
    cleaned = clean_text(combined)
    features = vectorizer.transform([cleaned])
    label = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    classes = list(model.classes_)
    fake_prob = float(proba[classes.index("FAKE")]) * 100
    real_prob = float(proba[classes.index("REAL")]) * 100
    return {
        "label": label,
        "fake_prob": round(fake_prob, 1),
        "real_prob": round(real_prob, 1),
        "confidence": round(max(fake_prob, real_prob), 1),
    }


# ── Test cases ────────────────────────────────────────────────────────
# expected: "REAL" | "FAKE" | None (None = no ground truth, log only)
TEST_CASES = [
    {
        "id": "real_1_city_council",
        "category": "real_factual",
        "expected": "REAL",
        "text": (
            "The city council voted 6-2 on Tuesday to approve a $4.2 million "
            "budget increase for road repairs, following months of public "
            "hearings on infrastructure conditions. Council members cited a "
            "recent engineering survey that found nearly a third of municipal "
            "roads in need of resurfacing within five years. The funding will "
            "be drawn from a combination of state grants and a small increase "
            "in the local vehicle registration fee, officials said."
        ),
    },
    {
        "id": "real_2_medical_study",
        "category": "real_factual",
        "expected": "REAL",
        "text": (
            "Researchers at a university medical center published findings "
            "this week showing a modest reduction in hospital readmission "
            "rates among patients enrolled in a post-discharge telehealth "
            "program. The study, which followed 1,200 patients over 18 months, "
            "found a 12 percent drop in 30-day readmissions compared to a "
            "control group. The authors cautioned that the sample size limits "
            "how broadly the results can be applied and called for larger, "
            "multi-site trials."
        ),
    },
    {
        "id": "real_3_earnings",
        "category": "real_factual",
        "expected": "REAL",
        "text": (
            "Shares of the retail chain fell nearly 8 percent in early trading "
            "after the company reported quarterly earnings that missed analyst "
            "expectations. Executives pointed to weaker foot traffic in "
            "suburban locations and higher shipping costs as the main drivers "
            "of the shortfall. The company said it still expects to meet its "
            "full-year revenue guidance, though several analysts revised their "
            "price targets downward following the call."
        ),
    },
    {
        "id": "real_4_election_audit",
        "category": "real_factual",
        "expected": "REAL",
        "text": (
            "Election officials in the county said Wednesday that a routine "
            "post-election audit of a sample of precincts found no "
            "discrepancies between machine counts and hand-counted ballots. "
            "The audit, required under state law for all general elections, "
            "covered roughly 5 percent of precincts selected at random. "
            "Results are expected to be certified by the county board next "
            "week."
        ),
    },
    {
        "id": "fake_5_obvious_caps",
        "category": "fake_obvious",
        "expected": "FAKE",
        "text": (
            "BREAKING: Leaked documents PROVE the water supply is being "
            "secretly treated with mind-altering compounds by unnamed "
            "government agents, according to an anonymous whistleblower. "
            "Sources say the cover-up goes back DECADES and mainstream "
            "scientists are being paid to stay silent. Share before this gets "
            "taken down!"
        ),
    },
    {
        "id": "fake_6_obvious_election",
        "category": "fake_obvious",
        "expected": "FAKE",
        "text": (
            "SHOCKING new evidence reveals that the recent election was "
            "rigged using a secret algorithm developed by tech insiders "
            "working with foreign governments. A former employee, speaking "
            "only on condition of anonymity, claims to have PROOF that will "
            "'destroy the official narrative' once released. The mainstream "
            "media refuses to cover this story."
        ),
    },
    {
        "id": "fake_7_subtle_vaccine",
        "category": "fake_subtle",
        "expected": "FAKE",
        "text": (
            "According to a document reportedly obtained by researchers, "
            "several major pharmaceutical companies have known for years "
            "that a common preservative in vaccines causes long-term "
            "neurological changes in children, but have suppressed this "
            "information through funding arrangements with regulatory "
            "bodies. The report has not been independently verified, and the "
            "companies named have not responded to requests for comment."
        ),
    },
    {
        "id": "fake_8_subtle_weather",
        "category": "fake_subtle",
        "expected": "FAKE",
        "text": (
            "A retired intelligence official says he has reviewed internal "
            "communications suggesting that a recent, widely reported "
            "natural disaster was exacerbated by a classified "
            "weather-modification program. He declined to name the agency "
            "involved, citing ongoing legal obligations, but said he felt "
            "compelled to speak out given what he described as a pattern of "
            "similar incidents in the region."
        ),
    },
    {
        "id": "ambig_9_satire",
        "category": "ambiguous",
        "expected": None,
        "text": (
            "Local man who read one article about interest rates now "
            "confidently explains monetary policy to strangers at dinner "
            "parties, sources close to the man's exhausted friend group "
            "confirm. 'He used the phrase quantitative easing four times last "
            "night,' said one attendee. 'Nobody asked.'"
        ),
    },
    {
        "id": "ambig_10_opinion",
        "category": "ambiguous",
        "expected": None,
        "text": (
            "It's long past time lawmakers stopped treating this issue as a "
            "partisan football and started treating it like the crisis it "
            "is. Every year of delay costs real people real money, and the "
            "excuses from both parties are wearing thin. Leadership means "
            "making hard calls before the next election cycle, not after."
        ),
    },
    {
        "id": "ambig_11_poor_sourcing",
        "category": "ambiguous",
        "expected": None,
        "text": (
            "A social media post claiming a major airline is quietly cutting "
            "safety inspections to save money has been viewed millions of "
            "times this week. The airline has denied the claim, calling it "
            "'categorically false,' but has not released inspection records "
            "publicly. Aviation regulators say they have no open "
            "investigation into the airline at this time."
        ),
    },
    {
        "id": "domain_12_non_us",
        "category": "domain_shift",
        "expected": "REAL",
        "text": (
            "The Reserve Bank raised its benchmark interest rate by 25 basis "
            "points on Thursday, citing persistent inflation in food and "
            "energy prices. The central bank's governor said further "
            "tightening 'cannot be ruled out' if price pressures do not ease "
            "in the coming quarters. The move was broadly anticipated by "
            "market analysts, though the size of the hike surprised some."
        ),
    },
]


# ── pytest fixtures + parametrized test ─────────────────────────────
@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["id"] for c in TEST_CASES])
def test_prediction_direction(case, artifacts):
    model, vectorizer = artifacts
    result = predict(case["text"], model, vectorizer)

    if case["expected"] is None:
        pytest.skip(f"{case['id']}: no ground truth, log-only (see printed table)")

    assert result["label"] == case["expected"], (
        f"{case['id']} ({case['category']}): expected {case['expected']}, "
        f"got {result['label']} (fake={result['fake_prob']}%, "
        f"real={result['real_prob']}%)"
    )


# ── standalone runner: prints a markdown table for the README ──────
def run_and_print_table():
    model, vectorizer = load_artifacts()
    print(f"| ID | Category | Expected | Predicted | Fake % | Real % | Confidence | Match |")
    print(f"|---|---|---|---|---|---|---|---|")

    n_correct, n_scored = 0, 0
    for case in TEST_CASES:
        r = predict(case["text"], model, vectorizer)
        expected = case["expected"] or "—"
        if case["expected"] is not None:
            n_scored += 1
            match = "✅" if r["label"] == case["expected"] else "❌"
            if match == "✅":
                n_correct += 1
        else:
            match = "—"
        print(
            f"| {case['id']} | {case['category']} | {expected} | {r['label']} "
            f"| {r['fake_prob']} | {r['real_prob']} | {r['confidence']} | {match} |"
        )

    print()
    if n_scored:
        print(f"Scored accuracy on labeled cases: {n_correct}/{n_scored} "
              f"({100 * n_correct / n_scored:.1f}%)")
    print("Ambiguous cases (satire, opinion, poor-sourcing) intentionally have "
          "no ground truth — log their predictions as documented limitations, "
          "not pass/fail results.")


if __name__ == "__main__":
    run_and_print_table()