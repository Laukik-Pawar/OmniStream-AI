
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


GENERIC_TERMS = {
    "peak", "guys", "finally", "just",
    "hope", "think", "ready", "incoming",
    "feature", "available", "day",
    "really", "damn", "thing", "things",
    "experience", "ama", "passed", "passing",
    "viral", "trending", "shorts"
}

STOPWORDS = set(ENGLISH_STOP_WORDS) | GENERIC_TERMS


# General search-intent rules.
# These are not user-specific interests.
INTENT_RULES = [
    (
        {"certification", "certified", "exam",
         "architect", "cloudops"},
        ["study guide", "practice questions", "tutorials"]
    ),
    (
        {"recipe", "recipes", "homemade",
         "cooking", "sandwich"},
        ["recipes", "cooking tips", "preparation"]
    ),
    (
        {"game", "gaming", "gameplay",
         "console", "ps5", "xbox"},
        ["gameplay", "updates", "guides"]
    ),
    (
        {"song", "songs", "music", "lyrics",
         "lyrical", "soundtrack"},
        ["similar songs", "music recommendations"]
    ),
    (
        {"pronunciation", "language",
         "grammar", "iast", "learning"},
        ["tutorials", "learning resources"]
    ),
    (
        {"motorcycle", "bike", "car",
         "vehicle", "scooter"},
        ["reviews", "comparisons", "owner experiences"]
    ),
    (
        {"food", "restaurant", "restaurants",
         "thali", "eating"},
        ["food recommendations", "local food guide"]
    ),
    (
        {"workout", "fitness", "cardio",
         "weightloss", "fatloss"},
        ["guides", "tips", "routines"]
    )
]


def clean_phrase(phrase):
    """Remove conversational noise from interest phrases."""

    phrase = str(phrase or "").lower()

    phrase = re.sub(r"https?://\S+", " ", phrase)
    phrase = re.sub(r"[^\w\s-]", " ", phrase)

    words = [
        word for word in phrase.split()
        if word not in STOPWORDS
    ]

    return " ".join(words).strip()


def get_tokens(text):
    return set(
        re.findall(r"(?u)\b[\w-]+\b", text.lower())
    )


def select_intents(profile):
    """Identify suitable discovery-oriented search intents."""

    text = " ".join(
        profile.get("interests", [])
        + profile.get("titles", [])
    )

    tokens = get_tokens(text)

    matches = []

    for triggers, intents in INTENT_RULES:

        overlap = len(tokens & triggers)

        if overlap:
            matches.append((overlap, intents))

    if not matches:
        return ["related content", "guide"]

    matches.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return matches[0][1]


def build_subject(profile, max_words=7):
    """Construct a subject from related interest phrases."""

    phrases = [
        profile.get("topic", "")
    ] + profile.get("interests", [])

    cleaned = []

    for phrase in phrases:

        phrase = clean_phrase(phrase)

        if phrase and phrase not in cleaned:
            cleaned.append(phrase)

    if not cleaned:
        return ""

    subject = cleaned[0]
    used_words = get_tokens(subject)

    # Add supporting phrases only when they share
    # meaningful context with the existing subject.
    for phrase in cleaned[1:]:

        phrase_words = get_tokens(phrase)

        if not phrase_words:
            continue

        if not used_words.intersection(phrase_words):
            continue

        new_words = phrase_words - used_words

        if not new_words:
            continue

        if len(subject.split()) + len(new_words) > max_words:
            continue

        additions = [
            word for word in phrase.split()
            if word not in used_words
        ]

        subject += " " + " ".join(additions)
        used_words.update(additions)

    return subject.strip()


def is_similar_query(query, existing_queries):
    """Prevent highly overlapping search queries."""

    words = get_tokens(query)

    for existing in existing_queries:

        existing_words = get_tokens(existing)

        union = words | existing_words

        if not union:
            continue

        similarity = len(
            words & existing_words
        ) / len(union)

        if similarity >= 0.75:
            return True

    return False


def generate_queries(interest_profiles, max_queries=5):
    """
    Generate personalized discovery queries from
    ranked interest profiles.
    """

    ranked_profiles = sorted(
        interest_profiles,
        key=lambda profile: (
            profile.get("preference_score", 0),
            profile.get("count", 0)
        ),
        reverse=True
    )

    queries = []
    seen_queries = []

    # Generate one query per interest group
    # before generating additional variants.
    for profile in ranked_profiles:

        subject = build_subject(profile)

        if not subject:
            continue

        # Skip weak, single-word subjects unless
        # supported by multiple interactions.
        if (
            len(subject.split()) < 2
            and profile.get("count", 0) < 2
        ):
            continue

        intents = select_intents(profile)

        for intent in intents:

            intent_words = get_tokens(intent)
            subject_words = get_tokens(subject)

            if intent_words.issubset(subject_words):
                query = subject
            else:
                query = f"{subject} {intent}"

            query = re.sub(
                r"\s+", " ", query
            ).strip()

            if is_similar_query(query, seen_queries):
                continue

            queries.append({
                "query": query,
                "keywords": subject.split(),
                "topic": profile.get("topic", ""),
                "interests": profile.get("interests", []),
                "sources": profile.get("sources", []),
                "preference_score": profile.get(
                    "preference_score", 0
                ),
                "intent": intent,
                "original_titles": profile.get("titles", [])
            })

            seen_queries.append(query)

            # One query per group for diversity.
            break

        if len(queries) >= max_queries:
            break

    return queries
