# Recommendation engine

OmniStream-AI now has a separate hybrid ranking layer in
`app/services/recommendation_engine.py`.

## Request flow

1. Reddit and YouTube history is collected.
2. The existing TF-IDF/K-Means pipeline groups history and creates search queries.
3. Google Custom Search retrieves candidates with SafeSearch enabled.
4. Invalid/non-HTTP links and duplicate canonical URLs are removed.
5. Candidate articles are ranked against the user's history.
6. The multilingual embedding model is loaded lazily; TF-IDF cosine similarity is the fallback if model loading fails.

## Ranking

The current transparent heuristic combines:

- **65% relevance** — multilingual semantic similarity blended with lexical TF-IDF similarity (75/25) when embeddings work; TF-IDF alone when they do not.
- **15% genre match** — whether the result's genre occurs in the user's history.
- **10% freshness** — a 30-day exponential half-life when a publication timestamp is available; neutral when it is missing.
- **10% quality** — a normalized supplied quality/search score, neutral by default.

These weights are an initial baseline, not trained or validated weights. Do not describe them as measured improvements until a reviewed evaluation set confirms it. Recommended offline metrics are Precision@K and nDCG@K, with genre accuracy, duplicate rate, invalid URL rate, and p95 latency tracked separately.

## Model configuration

Set `OMNISTREAM_EMBEDDING_MODEL` to override the default
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model. The model is fetched from Hugging Face on first semantic-ranking use and cached under `HF_HOME`. Ensure the deployed environment can reach Hugging Face or pre-populate the model cache. If loading fails, the service falls back to TF-IDF.

## Tests

Run the focused ranking tests from the repository root:

```bash
python -m unittest tests.test_recommendation_engine
```

The tests mock embedding failures to exercise the deterministic TF-IDF fallback without downloading a model.
