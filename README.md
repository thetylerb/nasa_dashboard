# NASA Data Pipeline & AI Analytics

A portfolio project combining **ETL engineering**, **NLP/ML modelling**, and **interactive data visualisation** — built entirely on free public NASA APIs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Sources                            │
│   NASA APOD API (api.nasa.gov)   NASA EONET API (eonet.gsfc.nasa.gov) │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
          ┌────▼────┐                ┌────▼────┐
          │ APOD    │                │ EONET   │
          │ingest   │                │ ingest  │
          └────┬────┘                └────┬────┘
               │                          │
          ┌────▼────┐                ┌────▼────┐
          │ APOD    │                │ EONET   │
          │transform│                │transform│
          └────┬────┘                └────┬────┘
               │                          │
               └──────────┬───────────────┘
                           │
                    ┌──────▼──────┐
                    │  SQLite DB  │
                    │  (data/)    │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                  │
    ┌────▼────┐      ┌─────▼────┐      ┌─────▼─────┐
    │Zero-shot│      │Sentiment │      │ Isolation │
    │Classify │      │& Urgency │      │  Forest   │
    │(APOD)   │      │(APOD)    │      │ (EONET)   │
    └────┬────┘      └─────┬────┘      └─────┬─────┘
         │                 │                  │
    ┌────▼────────────────▼──────────────────▼─────┐
    │              SQLite DB (ML results)           │
    └───────────────────────┬───────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │   Streamlit Dashboard       │
              │  ┌────────────────────┐    │
              │  │ Live Events Map    │    │
              │  │ APOD Explorer      │    │
              │  │ Trend Analysis     │    │
              │  │ Anomaly Report     │    │
              │  │ Semantic Search    │    │
              │  └────────────────────┘    │
              └────────────────────────────┘
```

### Folder structure

```
automated_etl_pipeline/
├── config/
│   └── config.py          # All configurable parameters
├── ingestion/
│   ├── apod_ingestion.py  # APOD API client (backfill + incremental)
│   └── eonet_ingestion.py # EONET API client (backfill + incremental)
├── transform/
│   ├── apod_transform.py  # Normalize raw APOD responses
│   └── eonet_transform.py # Normalize raw EONET responses + geometry selection
├── storage/
│   └── database.py        # SQLite schema, upsert helpers, query helpers
├── models/
│   ├── classifier.py      # Zero-shot classification (bart-large-mnli)
│   ├── sentiment.py       # Sentiment + urgency scoring (distilbert)
│   ├── anomaly.py         # Isolation Forest on EONET event frequency
│   └── semantic_search.py # Sentence-transformer embeddings + cosine search
├── dashboard/
│   ├── app.py             # Streamlit entry point + sidebar
│   └── pages/
│       ├── 1_Live_Events_Map.py
│       ├── 2_APOD_Explorer.py
│       ├── 3_Trend_Analysis.py
│       ├── 4_Anomaly_Report.py
│       └── 5_Semantic_Search.py
├── pipeline.py            # Orchestrator (backfill / incremental / ml-only)
├── scheduler.py           # APScheduler (EONET every 6h, APOD daily)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd automated_etl_pipeline

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Get a NASA API key

1. Visit **https://api.nasa.gov**
2. Fill in your name and email address — the key is emailed instantly and is free.
3. Copy `.env.example` to `.env` and paste your key:

```bash
cp .env.example .env
# Edit .env:
NASA_API_KEY=your_key_here
```

> **Note:** The `DEMO_KEY` default works for low-volume testing but is rate-limited to 30 requests/hour. The full APOD backfill (11 000+ entries, ~115 batched requests) requires a real key.

### 3. First run — full backfill

```bash
python pipeline.py --mode backfill
```

This will:
- Fetch the complete APOD archive (June 1995 → today, ~11 000 entries)
- Fetch 2 years of EONET events
- Run zero-shot classification on all APOD explanations *(takes 1–4 hours on CPU)*
- Score sentiment for Earth-related entries
- Generate sentence embeddings for semantic search
- Run anomaly detection on EONET event frequency

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

### 5. Schedule recurring updates

```bash
python scheduler.py
```

Runs silently in the background; EONET refreshes every 6 hours, APOD once daily.

---

## ML Components

### 1 — Zero-shot Text Classification (`facebook/bart-large-mnli`)

**Why chosen:** BART fine-tuned on MultiNLI can classify text into arbitrary label sets without any task-specific training data. This is ideal for the APOD archive: we have ~11 000 diverse astronomy descriptions but no labelled training set. The model frames each label as a natural-language hypothesis and scores the probability that the explanation *entails* it.

**Labels used:** Solar System · Deep Space · Earth Observation · Stars and Nebulae · Human Spaceflight · Atmospheric Phenomena

**Trade-off:** Slow on CPU (~0.5–2s/entry). Results are batched and cached in SQLite so the cost is paid once.

### 2 — Sentiment & Urgency Scoring (`distilbert-base-uncased-finetuned-sst-2-english`)

**Why chosen:** A lightweight (66M parameter) distilled BERT model, fast enough to run on CPU in bulk. Applied only to Earth-related APOD entries (classifications that map to observable phenomena: *Earth Observation*, *Atmospheric Phenomena*), giving a meaningful signal for tracking how the tone of science communication around Earth has shifted over 30 years.

Urgency is a keyword-density heuristic over climate/disaster vocabulary — designed to complement the binary sentiment score.

### 3 — Anomaly Detection on Event Frequency (`sklearn.IsolationForest`)

**Why chosen:** Isolation Forest is an ensemble tree method that explicitly models anomalies rather than fitting a distribution to normal data. It works well on small, univariate time-series features (rolling 30-day event counts), requires no labelled anomalies, and runs in milliseconds.

Each EONET category gets its own model. Windows with predicted class `-1` are flagged with a confidence score derived from the decision function value.

### 4 — Semantic Search (`all-MiniLM-L6-v2` via sentence-transformers)

**Why chosen:** `all-MiniLM-L6-v2` is a highly efficient sentence embedding model (22M parameters) that produces 384-dimensional embeddings optimised for semantic similarity. It outperforms keyword search for astronomy text, where the same concept appears with very different vocabulary (e.g., "neutron star", "pulsar", "stellar remnant").

Embeddings are stored as binary blobs in SQLite and loaded into memory at dashboard startup. Query time is O(n) cosine similarity — fast enough for 11 000 entries.

---

## Running Modes

| Command | What it does |
|---------|-------------|
| `python pipeline.py --mode backfill` | Full archive fetch + all ML (first run) |
| `python pipeline.py --mode incremental` | Fetch new entries since last run + ML on new data |
| `python pipeline.py --mode ml-only` | Re-run all ML on existing DB data (no API calls) |
| `python scheduler.py` | Start the background scheduler |
| `streamlit run dashboard/app.py` | Launch the interactive dashboard |

---

## Example Screenshots (described)

**Live Events Map** — A dark-themed world map with coloured dots: red clusters in the Gulf of Mexico (hurricane season), orange streaks across California and Australia (wildfires), blue markings near Greenland (sea ice events). Clicking a dot shows the event title, date, and magnitude in a tooltip.

**APOD Explorer** — A side-by-side layout: the famous "Pale Blue Dot" image on the left; on the right, the title, an AI label badge reading `Deep Space`, a green sentiment chip reading `POSITIVE 94%`, and a low urgency bar. Below, the full Sagan-authored explanation text.

**Trend Analysis** — A stacked bar chart showing how "Solar System" (blue) dominated APOD in the late 1990s, while "Earth Observation" (green) and "Stars and Nebulae" (purple) grew steadily through the 2010s and 2020s. Below it, EONET monthly event counts showing a clear uptick in severe storms from 2017 onward.

**Anomaly Report** — A timeline with a thin green line of rolling 30-day storm counts, punctuated by large red X markers at August–September 2017 (Hurricane Harvey/Irma/Maria), September 2022 (Hurricane Ian), and early 2024. A table below lists each flagged window with confidence scores above 80%.

**Semantic Search** — A query "eruption of a massive volcano with ash cloud" returns, in order: Mount Pinatubo (1991), Eyjafjallajökull (2010), Kilauea (2018), Sarychev Peak (2009), and a 2022 Tonga eruption entry — each with a similarity percentage and a highlighted snippet.

---

## What I Learned

Building this project forced me to think carefully about the full lifecycle of a data system, not just the modelling layer.

On the **ETL side**, the biggest lesson was designing for incremental updates from day one. I initially sketched the ingestion modules as simple "fetch everything" functions, then realised that re-fetching the entire APOD archive on every run (11 000+ API requests) was both wasteful and fragile. Rebuilding around chunked date ranges and upsert semantics made the pipeline idempotent — you can run it multiple times safely, which is critical for anything running on a schedule.

The **zero-shot classifier** was a revelation. I expected to need a labelled training set, which would have taken days to build. Instead, `facebook/bart-large-mnli` produced sensible classifications on the first pass with no training at all. The trade-off is speed: on CPU, classifying ~11 000 entries takes hours. Batching the API calls and caching results in SQLite was the key design choice that made this practical.

The **Isolation Forest** anomaly detection taught me to think carefully about what "normal" means in a time series. Applying a single model to all EONET data initially produced misleading results because storm frequency and earthquake frequency operate at completely different baselines. Grouping by category first and fitting separate models gave much more interpretable flags.

The **semantic search** component surprised me with how well it worked out of the box. Returning genuinely related APOD entries for free-text queries like "ice on another planet" — without any keyword overlap — made the dataset feel alive in a way that a simple date-based browser doesn't.

If I were extending this project, I would: (1) replace SQLite with PostgreSQL + pgvector for scalable similarity search, (2) add a proper vector index (HNSW or FAISS) to keep search fast as the corpus grows, and (3) explore fine-tuning a small LLM on APOD explanations to generate summaries rather than just classifying them.

---

## API References

- **APOD:** https://api.nasa.gov/planetary/apod
- **EONET:** https://eonet.gsfc.nasa.gov/docs/v3
- **API key registration:** https://api.nasa.gov
