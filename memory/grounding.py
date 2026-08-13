"""Deterministic check that a question's subject actually appears in the sources.

The model cannot be trusted to notice that "Software Engineer Manager" is not
the same role as "Senior Software Engineer" — it accepts the question's framing
and answers from the near-match. Prompt instructions do not fix this because the
faulty faculty is the comparison itself. So compare in code instead.
"""
import re

# capitalised words that carry no identity on their own
_GENERIC = {
    "I", "I'm", "I've", "My", "Me", "We", "The", "A", "An", "Do", "Does", "Did",
    "Can", "Could", "Would", "Will", "What", "When", "Where", "Who", "How", "Why",
    "Is", "Are", "Was", "Were", "Have", "Has", "Had", "Please", "Also", "And",
    "But", "So", "Then", "That", "This", "It", "You", "Your", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}


def _normalise(text: str) -> str:
    # collapse runs of spaces: punctuation becomes a space, so "Dr. Lee" must
    # end up identical to the "Dr Lee" the question yields after stripping dots
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def question_phrases(question: str) -> list[str]:
    """Multi-word capitalised phrases — the specific things being asked about."""
    words = question.split()
    phrases, run = [], []
    for i, raw in enumerate(words):
        word = raw.strip(".,!?;:'\"()")
        capitalised = bool(word) and word[0].isupper()
        # a sentence-initial capital says nothing about the word
        if capitalised and (i == 0 or word in _GENERIC):
            capitalised = False
        if capitalised:
            run.append(word)
        else:
            if len(run) >= 2:
                phrases.append(" ".join(run))
            run = []
    if len(run) >= 2:
        phrases.append(" ".join(run))
    return phrases


def unverified_terms(question: str, sources: list[str]) -> list[str]:
    """Phrases the question treats as given that appear in no retrieved source.

    Only multi-word proper phrases are checked: they are specific enough that
    absence is meaningful, which keeps false positives low.
    """
    phrases = question_phrases(question)
    if not phrases:
        return []
    # pad both sides so a phrase only matches whole words ("New York" must not
    # be satisfied by "New Yorker")
    haystack = f" {_normalise(' '.join(sources))} "
    return [p for p in phrases if f" {_normalise(p)} " not in haystack]
