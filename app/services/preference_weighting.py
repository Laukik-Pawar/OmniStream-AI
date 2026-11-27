
from datetime import datetime, timezone, timedelta
from collections import Counter


RECENT_DAYS = 7
HISTORY_DAYS = 30

RECENT_WEIGHT = 0.7
LONG_TERM_WEIGHT = 0.3


def parse_timestamp(value):
    """Convert supported timestamps to UTC datetime."""

    if value is None:
        return None

    try:
        if isinstance(value, datetime):
            dt = value

        elif isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(
                value,
                tz=timezone.utc
            )

        else:
            dt = datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc)

    except (ValueError, TypeError, OverflowError, OSError):
        return None


def weight_preferences(
    topic_groups,
    interest_profiles,
    allow_timestamp_proxy=True
):
    """
    Rank interest profiles using recent and
    long-term interaction distributions.

    Existing timestamps may represent observation
    or publication time rather than interaction time.
    """

    now = datetime.now(timezone.utc)

    recent_cutoff = now - timedelta(days=RECENT_DAYS)
    history_cutoff = now - timedelta(days=HISTORY_DAYS)

    recent_counts = Counter()
    long_term_counts = Counter()
    frequency_counts = Counter()

    proxy_count = 0
    unknown_count = 0
    outside_window_count = 0

    def group_key(titles):
        return tuple(sorted(str(t or "") for t in titles))

    # 1. Count interactions in each time window.
    for group in topic_groups:

        key = group_key([
            item.get("title", "")
            for item in group["interactions"]
        ])

        for item in group["interactions"]:

            frequency_counts[key] += 1

            timestamp = parse_timestamp(
                item.get("interaction_timestamp")
            )

            # Temporary proxy until reliable
            # interaction timestamps are available.
            if timestamp is None and allow_timestamp_proxy:

                timestamp = parse_timestamp(
                    item.get("timestamp")
                )

                if timestamp is not None:
                    proxy_count += 1

            if timestamp is None or timestamp > now:
                unknown_count += 1
                continue

            if timestamp >= recent_cutoff:
                recent_counts[key] += 1

            elif timestamp >= history_cutoff:
                long_term_counts[key] += 1

            else:
                outside_window_count += 1

    total_recent = sum(recent_counts.values())
    total_long_term = sum(long_term_counts.values())
    total_interactions = sum(frequency_counts.values())

    # 2. Handle empty time windows.
    if total_recent and total_long_term:

        recent_weight = RECENT_WEIGHT
        long_term_weight = LONG_TERM_WEIGHT
        mode = "70_30"

    elif total_recent:

        recent_weight = 1.0
        long_term_weight = 0.0
        mode = "recent_only"

    elif total_long_term:

        recent_weight = 0.0
        long_term_weight = 1.0
        mode = "long_term_only"

    else:

        recent_weight = 0.0
        long_term_weight = 0.0
        mode = "frequency_fallback"

    weighted_profiles = []

    # 3. Calculate the preference score.
    for profile in interest_profiles:

        key = group_key(profile.get("titles", []))

        recent_count = recent_counts[key]
        long_term_count = long_term_counts[key]

        recent_score = (
            recent_count / total_recent
            if total_recent else 0.0
        )

        long_term_score = (
            long_term_count / total_long_term
            if total_long_term else 0.0
        )

        if mode == "frequency_fallback":

            preference_score = (
                frequency_counts[key] / total_interactions
                if total_interactions else 0.0
            )

        else:

            preference_score = (
                recent_weight * recent_score
                + long_term_weight * long_term_score
            )

        weighted_profiles.append({
            **profile,
            "recent_count": recent_count,
            "long_term_count": long_term_count,
            "recent_score": round(recent_score, 4),
            "long_term_score": round(long_term_score, 4),
            "preference_score": round(preference_score, 6)
        })

    # 4. Prioritize stronger interests.
    weighted_profiles.sort(
        key=lambda profile: (
            profile["preference_score"],
            profile["count"]
        ),
        reverse=True
    )

    # 5. Return profiles and debugging information.
    return {
        "profiles": weighted_profiles,
        "summary": {
            "mode": mode,
            "recent_days": RECENT_DAYS,
            "history_days": HISTORY_DAYS,
            "recent_weight": recent_weight,
            "long_term_weight": long_term_weight,
            "recent_interactions": total_recent,
            "long_term_interactions": total_long_term,
            "total_interactions": total_interactions,
            "proxy_timestamp_count": proxy_count,
            "unknown_timestamp_count": unknown_count,
            "outside_window_count": outside_window_count,
            "timestamp_basis": (
                "includes_timestamp_proxies"
                if proxy_count
                else "interaction_timestamps_only"
            )
        }
    }
