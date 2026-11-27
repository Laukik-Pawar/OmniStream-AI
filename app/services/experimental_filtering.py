
import re
import unicodedata

from difflib import SequenceMatcher
from urllib.parse import (
    urlsplit,
    parse_qs,
    parse_qsl,
    urlencode
)


TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "dclid",
    "msclkid",
    "igshid",
    "mc_cid",
    "mc_eid"
}

NEAR_DUPLICATE_THRESHOLD = 0.95


def normalize_url(url):
    """
    Convert different URL representations of the
    same content into a canonical identity.
    """

    if not url:
        return ""

    url = str(url).strip()

    if url.startswith("/r/"):
        url = "https://www.reddit.com" + url

    if url.startswith("//"):
        url = "https:" + url

    if "://" not in url:
        url = "https://" + url

    try:
        parsed = urlsplit(url)

        host = (parsed.hostname or "").lower()

        if host.startswith("www."):
            host = host[4:]

        path = parsed.path
        query = parse_qs(parsed.query)

        # 1. YouTube content identity.
        if (
            host == "youtube.com"
            or host.endswith(".youtube.com")
            or host == "youtu.be"
        ):

            video_id = None

            if host == "youtu.be":
                video_id = path.strip("/").split("/")[0]

            elif path == "/watch":
                video_id = query.get("v", [None])[0]

            else:
                match = re.match(
                    r"/(?:shorts|embed|live)/([^/]+)",
                    path
                )

                if match:
                    video_id = match.group(1)

            if video_id:
                return f"youtube:{video_id}"

        # 2. Reddit content identity.
        if (
            host == "reddit.com"
            or host.endswith(".reddit.com")
            or host == "redd.it"
        ):

            post_id = None

            if host == "redd.it":
                post_id = path.strip("/").split("/")[0]

            else:
                match = re.search(
                    r"/comments/([a-zA-Z0-9]+)",
                    path
                )

                if match:
                    post_id = match.group(1)

            if post_id:
                return f"reddit:{post_id.lower()}"

        # 3. General website URL normalization.
        parameters = []

        for key, value in parse_qsl(
            parsed.query,
            keep_blank_values=True
        ):

            if key.lower().startswith("utm_"):
                continue

            if key.lower() in TRACKING_PARAMETERS:
                continue

            parameters.append((key, value))

        parameters.sort()

        normalized_query = urlencode(parameters)

        normalized_path = path.rstrip("/")

        identity = f"{host}{normalized_path}"

        if normalized_query:
            identity += f"?{normalized_query}"

        return identity if host else ""

    except (ValueError, TypeError):
        return ""


def normalize_title(title):
    """
    Normalize titles while preserving multilingual text.
    """

    if not title:
        return ""

    title = unicodedata.normalize(
        "NFKC",
        str(title)
    ).casefold()

    # Remove common search-result platform suffixes.
    title = re.sub(
        r"\s*[-–—|]\s*(youtube|reddit)\s*$",
        "",
        title
    )

    # Remove subreddit suffixes.
    title = re.sub(
        r"\s*:\s*r/[\w]+\s*$",
        "",
        title
    )

    # Preserve Unicode letters and combining marks.
    title = "".join(
        char
        if (
            char.isalnum()
            or unicodedata.category(char).startswith("M")
        )
        else " "
        for char in title
    )

    return " ".join(title.split())


def get_content_identity(item):
    """
    Extract a canonical identity from available URL fields.
    """

    for field in ("url", "permalink", "link"):

        identity = normalize_url(item.get(field))

        if identity:
            return identity

    return ""


def is_near_duplicate(title_a, title_b):
    """
    Conservative title-based duplicate detection.
    """

    if not title_a or not title_b:
        return False

    # Avoid fuzzy matching short or generic titles.
    if min(len(title_a), len(title_b)) < 30:
        return False

    similarity = SequenceMatcher(
        None,
        title_a,
        title_b,
        autojunk=False
    ).ratio()

    return similarity >= NEAR_DUPLICATE_THRESHOLD


def filter_candidates(candidates, history):
    """
    Remove previously interacted content and duplicates.

    Candidates are expected to be semantically ranked.
    The highest-ranked version of duplicate content
    is retained.
    """

    candidates = candidates or []
    history = history or []

    consumed_urls = set()
    consumed_titles = set()

    # 1. Build previously interacted content identities.
    for item in history:

        identity = get_content_identity(item)

        title = normalize_title(item.get("title"))

        if identity:
            consumed_urls.add(identity)

        if title:
            consumed_titles.add(title)

    # 2. Initialize filtering state.
    seen_urls = set()
    seen_titles = set()

    filtered_candidates = []

    summary = {
        "input_candidates": len(candidates),
        "history_interactions": len(history),
        "history_url_identities": len(consumed_urls),
        "history_title_identities": len(consumed_titles),
        "removed_consumed_url": 0,
        "removed_consumed_title": 0,
        "removed_duplicate_url": 0,
        "removed_duplicate_title": 0,
        "removed_near_duplicate": 0,
        "removed_invalid_url": 0
    }

    removed_candidates = []

    # 3. Process candidates in personalized ranking order.
    for candidate in candidates:

        item = dict(candidate)

        identity = get_content_identity(item)

        title = normalize_title(item.get("title"))

        reason = None

        if not identity:
            reason = "invalid_url"

        elif identity in consumed_urls:
            reason = "consumed_url"

        elif title and title in consumed_titles:
            reason = "consumed_title"

        elif identity in seen_urls:
            reason = "duplicate_url"

        elif title and title in seen_titles:
            reason = "duplicate_title"

        elif any(
            is_near_duplicate(title, previous_title)
            for previous_title in seen_titles
        ):
            reason = "near_duplicate"

        # Record rejected content for debugging.
        if reason:

            summary[f"removed_{reason}"] += 1

            removed_candidates.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "reason": reason,
                "final_score": item.get("final_score")
            })

            continue

        # Retain unique content.
        seen_urls.add(identity)

        if title:
            seen_titles.add(title)

        item["canonical_identity"] = identity

        item["pre_filter_rank"] = item.get(
            "personalized_rank"
        )

        filtered_candidates.append(item)

    # 4. Reassign ranking positions after filtering.
    for rank, item in enumerate(
        filtered_candidates,
        start=1
    ):
        item["personalized_rank"] = rank

    summary["output_candidates"] = len(
        filtered_candidates
    )

    summary["total_removed"] = (
        len(candidates) - len(filtered_candidates)
    )

    summary["retention_rate"] = round(
        len(filtered_candidates) / len(candidates),
        4
    ) if candidates else 0.0

    return {
        "candidates": filtered_candidates,
        "summary": summary,
        "removed_candidates": removed_candidates
    }
