"""
test_predictions.py — Regression suite for the fake-news detector.

Uses paragraph-length test cases (not bare headlines) since the model is
trained on title x2 + full article body, and bare headlines produce
out-of-distribution TF-IDF vectors (see Session Summary, Step 3 finding).

Covers English (model.pkl/vectorizer.pkl) and Hindi (model_hi.pkl/
vectorizer_hi.pkl) separately — each language's cases are scored against
its own model, since the two are entirely separate classifiers.

NOTE ON THE HINDI TEST CASES: drafted by an LLM, not a fluent/native
Hindi speaker on the team. They have NOT been independently verified for
factual accuracy, natural phrasing, or correct REAL/FAKE labeling.
Review and correct before trusting these results — see README's Known
Limitations if these remain unverified.

Two modes:
  1. pytest test_predictions.py          -> soft assertions, CI-friendly
  2. python test_predictions.py          -> prints a markdown results table
     you can paste straight into the README's "known limitations" section
     (print-only; run pytest separately to enforce regression checks)

Soft assertions check *direction* (label) not exact confidence, since RF
output will drift a few points across retrains even with the same data.
"""

import os
import pickle
import sys

try:
    import pytest
except ImportError:
    pytest = None  # standalone `python test_predictions.py` mode doesn't need it

# ── Locate model artifacts ───────────────────────────────────────────
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
    sys.path.insert(0, os.path.join(HERE, "backend"))
    from src.preprocess import clean_text  # type: ignore


def _find_file(filename):
    for d in CANDIDATE_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None  # caller decides whether missing is fatal or skippable


def load_artifacts(lang="en"):
    """
    Load (model, vectorizer) for the given language. English keeps the
    unsuffixed filenames (model.pkl/vectorizer.pkl); every other
    language uses model_<lang>.pkl/vectorizer_<lang>.pkl, matching
    train.py's _artifact_names() convention. Returns (None, None) if
    either file is missing, rather than raising — lets callers skip a
    language cleanly instead of crashing the whole suite.
    """
    vec_name = "vectorizer.pkl" if lang == "en" else f"vectorizer_{lang}.pkl"
    model_name = "model.pkl" if lang == "en" else f"model_{lang}.pkl"

    vec_path = _find_file(vec_name)
    model_path = _find_file(model_name)

    if vec_path is None or model_path is None:
        return None, None

    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model, vectorizer


def predict(text, model, vectorizer, title="", lang="en"):
    combined = title + " " + title + " " + text
    cleaned = clean_text(combined, lang=lang)
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


# ── English test cases ───────────────────────────────────────────────
# expected: "REAL" | "FAKE" | None (None = no ground truth, log only)
EN_TEST_CASES = [
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


# ── Hindi test cases ─────────────────────────────────────────────────
# DRAFT ONLY — needs native/fluent review before these results are
# trusted. Structurally mirrors the English set (real/fake/ambiguous)
# but phrasing, naturalness, and factual-framing accuracy in Hindi have
# not been verified by anyone fluent. Treat scores from this block as
# provisional until reviewed.
HI_TEST_CASES = [
    {
        "id": "hi_real_1_civic",
        "category": "real_factual",
        "expected": "REAL",
        "text": (
            "नगर निगम ने मंगलवार को आयोजित बैठक में सड़क मरम्मत के लिए बजट "
            "बढ़ाने को औपचारिक स्वीकृति दे दी है। अधिकारियों ने बताया कि हाल ही में "
            "कराए गए सर्वेक्षण के अनुसार शहर की करीब एक तिहाई सड़कों को अगले पांच "
            "वर्षों में मरम्मत की आवश्यकता है। इस कार्य के लिए राज्य सरकार से मिलने "
            "वाली सहायता राशि और वाहन पंजीकरण शुल्क में आंशिक वृद्धि से प्राप्त "
            "राजस्व का उपयोग किया जाएगा।"
        ),
    },
    {
        "id": "hi_real_2_earnings",
        "category": "real_factual",
        "expected": "REAL",
        "text": (
            "एक प्रमुख खुदरा कंपनी के शेयरों में गुरुवार को करीब आठ प्रतिशत "
            "की गिरावट दर्ज की गई, क्योंकि तिमाही वित्तीय नतीजे विश्लेषकों के अनुमान "
            "से कमजोर रहे। कंपनी के प्रबंधन ने इसके लिए ग्राहकों की कम संख्या "
            "और बढ़ती परिवहन लागत को जिम्मेदार ठहराया। हालांकि, कंपनी ने चालू वित्तीय "
            "वर्ष के अपने निर्धारित लक्ष्यों को हासिल करने की उम्मीद जताई है।"
        ),
    },
    {
        "id": "hi_fake_3_obvious",
        "category": "fake_obvious",
        "expected": "FAKE",
        "text": (
            "बड़ा खुलासा! गुप्त दस्तावेजों से यह बात सामने आई है कि पानी की सप्लाई "
            "में चुपचाप माइंड-कंट्रोल (दिमाग को नियंत्रित करने वाले) रसायन मिलाए जा "
            "रहे हैं। एक अंदरूनी सूत्र के मुताबिक, यह खौफनाक साजिश वर्षों से चल रही "
            "है और मुख्यधारा के वैज्ञानिकों को इस पर मुंह बंद रखने के लिए भारी रकम "
            "दी जा रही है। इसे डिलीट किए जाने से पहले तुरंत शेयर करें!"
        ),
    },
    {
        "id": "hi_fake_4_subtle",
        "category": "fake_subtle",
        "expected": "FAKE",
        "text": (
            "सामने आए कुछ गोपनीय दस्तावेजों के अनुसार, कई बड़ी दवा निर्माता कंपनियों "
            "को यह बात पहले से ज्ञात थी कि टीकों में प्रयुक्त होने वाला एक आम घटक "
            "बच्चों में दीर्घकालिक तंत्रिका संबंधी विकार उत्पन्न करता है। हालांकि, "
            "नियामक निकायों के साथ अंदरूनी समझौतों के माध्यम से इस तथ्य को छिपाया "
            "गया। इस तथाकथित रिपोर्ट की पुष्टि किसी स्वतंत्र स्रोत द्वारा नहीं की गई है।"
        ),
    },
    {
        "id": "hi_ambig_5_opinion",
        "category": "ambiguous",
        "expected": None,
        "text": (
            "अब समय आ गया है कि राजनीतिक दल इस गंभीर मुद्दे पर बयानबाजी बंद करें "
            "और इसे एक वास्तविक संकट के रूप में लें। सुधारों में हर साल की देरी "
            "आम नागरिकों को भारी नुकसान पहुंचा रही है, और दोनों पक्षों की तरफ से "
            "दिए जाने वाले बहाने अब लोगों को स्वीकार्य नहीं हैं।"
        ),
    },
]


# ── pytest fixtures + parametrized test (English) ───────────────────
@pytest.fixture(scope="module") if pytest else (lambda f: f)
def en_artifacts():
    return load_artifacts(lang="en")


if pytest:
    @pytest.mark.parametrize("case", EN_TEST_CASES, ids=[c["id"] for c in EN_TEST_CASES])
    def test_prediction_direction_en(case, en_artifacts):
        model, vectorizer = en_artifacts
        if model is None:
            pytest.skip("English model artifacts not found — run main.py first")
        result = predict(case["text"], model, vectorizer, lang="en")

        if case["expected"] is None:
            pytest.skip(f"{case['id']}: no ground truth, log-only (see printed table)")

        assert result["label"] == case["expected"], (
            f"{case['id']} ({case['category']}): expected {case['expected']}, "
            f"got {result['label']} (fake={result['fake_prob']}%, "
            f"real={result['real_prob']}%)"
        )

    @pytest.fixture(scope="module")
    def hi_artifacts():
        return load_artifacts(lang="hi")

    @pytest.mark.parametrize("case", HI_TEST_CASES, ids=[c["id"] for c in HI_TEST_CASES])
    def test_prediction_direction_hi(case, hi_artifacts):
        model, vectorizer = hi_artifacts
        if model is None:
            pytest.skip("Hindi model artifacts not found — Hindi dataset may not be present")
        result = predict(case["text"], model, vectorizer, lang="hi")

        if case["expected"] is None:
            pytest.skip(f"{case['id']}: no ground truth, log-only (see printed table)")

        assert result["label"] == case["expected"], (
            f"{case['id']} ({case['category']}): expected {case['expected']}, "
            f"got {result['label']} (fake={result['fake_prob']}%, "
            f"real={result['real_prob']}%)"
        )


# ── standalone runner: prints a markdown table per language ─────────
def _run_language(lang, cases, label):
    model, vectorizer = load_artifacts(lang=lang)
    if model is None:
        print(f"\n[skip] {label}: model artifacts not found for lang='{lang}' — skipping.")
        return

    print(f"\n### {label} ({lang})\n")
    print(f"| ID | Category | Expected | Predicted | Fake % | Real % | Confidence | Match |")
    print(f"|---|---|---|---|---|---|---|---|")

    n_correct, n_scored = 0, 0
    for case in cases:
        r = predict(case["text"], model, vectorizer, lang=lang)
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
        print(f"Scored accuracy ({label}, {lang}): {n_correct}/{n_scored} "
              f"({100 * n_correct / n_scored:.1f}%)")


def run_and_print_table():
    _run_language("en", EN_TEST_CASES, "English")
    _run_language(
        "hi", HI_TEST_CASES,
        "Hindi (DRAFT test cases — unreviewed by a fluent speaker, treat as provisional)",
    )
    print(
        "\nAmbiguous cases (satire, opinion, poor-sourcing) intentionally have "
        "no ground truth — log their predictions as documented limitations, "
        "not pass/fail results."
    )


if __name__ == "__main__":
    run_and_print_table()
