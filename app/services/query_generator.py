
import re
import logging

import numpy as np

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from app.services.recommendation_engine import _embed_texts
from app.services.interest_extractor import (
    clean_interest_text,
    phrase_analyzer
)


logger = logging.getLogger(__name__)


NOISE_WORDS = {
    "peak", "guys", "finally", "just",
    "hope", "think", "ready", "incoming",
    "feature", "available", "day",
    "really", "damn", "thing", "things",
    "ama", "passed", "passing", "viral",
    "trending", "shorts", "genuine",
    "doubt", "read", "body", "does",
    "mean", "later", "reason", "actual",
    "properly", "w", "s", "m", "t",
    "try", "trash", "interesting",
    "subscribe", "follow", "comment"
}

STOPWORDS = set(ENGLISH_STOP_WORDS) | NOISE_WORDS


INTENT_RULES = [
    (
        {"certification", "certified", "exam",
         "architect", "cloudops"},
        "study guide"
    ),
    (
        {"recipe", "recipes", "homemade",
         "cooking", "sandwich", "masala"},
        "recipes"
    ),
    (
        {"game", "gaming", "gameplay",
         "console", "ps5", "xbox"},
        "gameplay updates"
    ),
    (
        {"song", "songs", "music", "lyrics",
         "lyrical", "soundtrack"},
        "music recommendations"
    ),
    (
        {"pronunciation", "language",
         "grammar", "iast", "transliteration",
         "transliterations"},
        "tutorials"
    ),
    (
        {"motorcycle", "bike", "car",
         "vehicle", "scooter"},
        "reviews"
    ),
    (
        {"food", "restaurant", "restaurants",
         "thali", "eating"},
        "food recommendations"
    ),
    (
        {"workout", "fitness", "cardio",
         "weightloss", "fatloss"},
        "guides"
    ),
    (
        {"fashion", "styling", "fashiontips",
         "ootd"},
        "style inspiration"
    ),
    (
        {"movie", "film", "cinema", "box office"},
        "updates"
    ),
    (
        {"park", "coaster", "theme park"},
        "visitor guide"
    )
]


def clean_phrase(phrase):

    phrase = str(phrase or "").lower()

    phrase = re.sub(
        r"https?://\S+",
        " ",
        phrase
    )

    phrase = re.sub(
        r"[^\w\s-]",
        " ",
        phrase
    )

    words = [
        word for word in phrase.split()
        if word not in STOPWORDS
        and len(word) > 1
    ]

    return " ".join(words).strip()


def get_tokens(text):

    return set(
        re.findall(
            r"(?u)\b[\w-]+\b",
            str(text).lower()
        )
    )


def group_key(titles):

    return tuple(
        sorted(str(title or "") for title in titles)
    )


def valid_subject(subject):

    words = subject.split()

    if len(words) < 2 or len(words) > 8:
        return False

    if all(word.isnumeric() for word in words):
        return False

    if len(set(words)) < 2:
        return False

    return True


def select_intent(subject, context):

    subject_tokens = get_tokens(subject)
    context_tokens = get_tokens(context)

    best_score = 0
    best_intent = ""

    for triggers, intent in INTENT_RULES:

        score = (
            3 * len(subject_tokens & triggers)
            + len(context_tokens & triggers)
        )

        if score > best_score:

            best_score = score
            best_intent = intent

    # Avoid meaningless generic suffixes.
    return best_intent


def build_candidates(profile, interactions):
    """
    Generate coherent candidate subjects from
    individual source interactions.
    """

    candidates = []
    seen = set()

    def add_candidate(phrase, context, source, origin):

        subject = clean_phrase(phrase)

        if not valid_subject(subject):
            return

        key = (subject, context)

        if key in seen:
            return

        seen.add(key)

        candidates.append({
            "subject": subject,
            "context": context,
            "source": source,
            "origin": origin
        })

    for item in interactions:

        title = clean_interest_text(
            item.get("title")
        )

        content = clean_interest_text(
            item.get("content")
        )[:400]

        subreddit = clean_interest_text(
            item.get("subreddit")
        )

        genre = clean_interest_text(
            item.get("native_genre")
        )

        source = item.get("source") or "unknown"

        context = " ".join(
            part for part in [
                title,
                content,
                subreddit,
                genre
            ]
            if part
        )

        if not context:
            continue

        title_tokens = get_tokens(title)

        # Use extracted phrases only when they
        # are supported by this individual title.
        for phrase in profile.get("interests", []):

            phrase_tokens = get_tokens(phrase)

            if (
                phrase_tokens
                and phrase_tokens.issubset(title_tokens)
            ):
                add_candidate(
                    phrase,
                    context,
                    source,
                    "interest"
                )

        # Generate phrases independently from
        # each title.
        title_phrases = sorted(
            set(phrase_analyzer(title)),
            key=lambda phrase: (
                len(phrase.split()),
                len(phrase)
            ),
            reverse=True
        )

        for phrase in title_phrases[:15]:

            add_candidate(
                phrase,
                context,
                source,
                "title"
            )

        # Recover context when titles are vague.
        content_phrases = sorted(
            set(phrase_analyzer(content)),
            key=lambda phrase: (
                len(phrase.split()),
                len(phrase)
            ),
            reverse=True
        )

        for phrase in content_phrases[:20]:

            add_candidate(
                phrase,
                context,
                source,
                "description"
            )

    return candidates


def rank_candidates(candidates):
    """
    Rank subjects using semantic relevance
    to their individual source contexts.
    """

    if not candidates:
        return []

    contexts = list(dict.fromkeys(
        item["context"]
        for item in candidates
    ))

    subjects = list(dict.fromkeys(
        item["subject"]
        for item in candidates
    ))

    context_index = {
        text: index
        for index, text in enumerate(contexts)
    }

    subject_index = {
        text: index
        for index, text in enumerate(subjects)
    }

    try:

        embeddings = _embed_texts(
            contexts + subjects
        )

        similarities = cosine_similarity(
            embeddings[len(contexts):],
            embeddings[:len(contexts)]
        )

        for item in candidates:

            semantic = float(
                np.clip(
                    similarities[
                        subject_index[item["subject"]],
                        context_index[item["context"]]
                    ],
                    0,
                    1
                )
            )

            specificity = min(
                len(item["subject"].split()) / 4,
                1.0
            )

            origin_bonus = {
                "interest": 0.10,
                "title": 0.08,
                "description": 0.0
            }.get(item["origin"], 0.0)

            item["score"] = (
                0.8 * semantic
                + 0.2 * specificity
                + origin_bonus
            )

            item["semantic_score"] = semantic

    except Exception:

        logger.exception(
            "Query candidate semantic scoring failed."
        )

        for item in candidates:

            item["score"] = (
                0.2 * min(
                    len(item["subject"].split()) / 4,
                    1.0
                )
                + (
                    0.1
                    if item["origin"] == "interest"
                    else 0.0
                )
            )

            item["semantic_score"] = None

    return sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True
    )


def is_similar_query(query, existing_queries):

    words = get_tokens(query)

    for existing in existing_queries:

        existing_words = get_tokens(existing)

        union = words | existing_words

        if not union:
            continue

        similarity = (
            len(words & existing_words)
            / len(union)
        )

        if similarity >= 0.75:
            return True

    return False


def generate_queries(
    interest_profiles,
    topic_groups,
    max_queries=5
):
    """
    Generate coherent discovery queries with
    preference ranking and platform diversity.
    """

    if max_queries <= 0:
        return []

    groups = {}

    for group in topic_groups:

        key = group_key([
            item.get("title", "")
            for item in group["interactions"]
        ])

        groups[key] = group["interactions"]

    ranked_profiles = sorted(
        interest_profiles,
        key=lambda profile: (
            profile.get("preference_score", 0),
            profile.get("count", 0)
        ),
        reverse=True
    )

    query_pool = []

    # 1. Generate candidates per interest group.
    for group_id, profile in enumerate(ranked_profiles):

        # Exclude profiles outside the active
        # preference windows.
        if profile.get("preference_score", 0) <= 0:
            continue

        key = group_key(
            profile.get("titles", [])
        )

        interactions = groups.get(key, [])

        candidates = build_candidates(
            profile,
            interactions
        )

        candidates = rank_candidates(candidates)

        # Keep a few alternatives for diversity.
        for candidate in candidates[:5]:

            semantic = candidate["semantic_score"]

            if semantic is not None and semantic < 0.35:
                continue

            subject = candidate["subject"]

            intent = select_intent(
                subject,
                candidate["context"]
            )

            if (
                intent
                and not get_tokens(intent).issubset(
                    get_tokens(subject)
                )
            ):
                query = f"{subject} {intent}"

            else:
                query = subject

            query_pool.append({
                "query": query,
                "keywords": subject.split(),
                "topic": profile.get("topic", ""),
                "interests": profile.get("interests", []),
                "sources": profile.get("sources", []),
                "source": candidate["source"],
                "preference_score": profile.get(
                    "preference_score", 0
                ),
                "query_quality": round(
                    candidate["score"], 4
                ),
                "semantic_score": (
                    round(semantic, 4)
                    if semantic is not None
                    else None
                ),
                "origin": candidate["origin"],
                "intent": intent,
                "original_titles": profile.get(
                    "titles", []
                ),
                "group_id": group_id
            })

    # 2. Prioritize preferences, then query quality.
    query_pool.sort(
        key=lambda item: (
            item["preference_score"],
            item["query_quality"]
        ),
        reverse=True
    )

    selected = []
    selected_groups = set()
    seen_queries = []

    def add_query(candidate):

        if len(selected) >= max_queries:
            return False

        if candidate["group_id"] in selected_groups:
            return False

        if is_similar_query(
            candidate["query"],
            seen_queries
        ):
            return False

        selected.append(candidate)

        selected_groups.add(
            candidate["group_id"]
        )

        seen_queries.append(
            candidate["query"]
        )

        return True

    # 3. Select the strongest overall query.
    if query_pool:

        add_query(query_pool[0])

    # 4. Reserve representation for both platforms
    # when eligible candidates are available.
    for platform in ["reddit", "youtube"]:

        if any(
            item["source"] == platform
            for item in selected
        ):
            continue

        for candidate in query_pool:

            if candidate["source"] != platform:
                continue

            if add_query(candidate):
                break

    # 5. Fill remaining slots by preference.
    for candidate in query_pool:

        if len(selected) >= max_queries:
            break

        add_query(candidate)

    selected.sort(
        key=lambda item: (
            item["preference_score"],
            item["query_quality"]
        ),
        reverse=True
    )

    return selected
