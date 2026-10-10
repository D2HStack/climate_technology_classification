# scraping_wayback_machine

# Wayback Machine Batch Scraper (`scrape_dataframe_in_batches`)

This script queries the Internet Archive’s Wayback Machine CDX API to retrieve, parse, and store historical webpage snapshots (both homepages and internal subpages) aligned around specific investment or event dates.

---

## What the Main Function Does

The primary entry point is `scrape_dataframe_in_batches`. It automates the end-to-end extraction workflow across a pandas `DataFrame`:

1. **State Recovery & Checkpointing:**
* Scans the target directory for existing checkpoint files matching the output filename pattern (e.g., `output_50.parquet`, `output_100.parquet`).
* Loads the latest checkpoint if available, allowing long-running scraping jobs to resume seamlessly where they left off.


2. **DataFrame Schema Initialization:**
* Expands the DataFrame into a flat wide schema (handled by `initialize_wide_dataframe`), pre-allocating dedicated columns for the landing page (`home_*`) and up to $N$ internal subpages (`subpage1_*` through `subpageN_*`).


3. **Time-Bounded Snapshot Retrieval:**
* For each row, calculates search windows around a target date (e.g., up to $M$ months before and/or after).
* Queries the Wayback Machine's CDX Server API via `fetch_single_wayback_snapshot` using exact and wildcard domain patterns.
* Pulls the nearest valid snapshot ($HTTP\ 200/301/302$) within the requested window.


4. **Content Extraction & Compression:**
* Fetches the snapshot's raw HTML using a pooled `requests.Session` configured with exponential backoff retries.
* Compresses the raw HTML using gzip and base64 encoding to keep file sizes compact.
* Extracts clean article text, metadata, page title, and word count using `trafilatura`.
* Optionally discovers internal first-order links using `BeautifulSoup` and scrapes up to `max_subpages` linked snapshots under the same base domain and timestamp.


5. **Batching, Checkpointing & Telemetry:**
* Processes rows in batches (default: 50).
* Periodically flushes progress to disk as compressed `.parquet` (using `zstd`) or `.pkl` files.
* Outputs runtime statistics, including average sample fetch duration, word counts, subpages scraped, and error distributions.



---

## Function Signature

```python
def scrape_dataframe_in_batches(
    df: pd.DataFrame,
    out_file: str,
    directory: str = "./",
    batch_size: int = 50,
    start_index: int | None = None,
    website_col: str = "investee_website",
    date_col: str = "investment_date",
    before_months: int = 0,
    after_months: int = 18,
    max_sample_time: float = 20.0,
    scrape_subpages: bool = True,
    max_subpages: int = 5,
    delay_range: tuple = (1.0, 2.0),
    timeout: float = 6.0,
    file_format: str = "parquet",
) -> pd.DataFrame

```

---

## Inputs

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `df` | `pd.DataFrame` | *Required* | Input DataFrame containing website URLs and associated event dates. |
| `out_file` | `str` | *Required* | Base name for the checkpointed output files (e.g., `"scraped_investees.parquet"`). |
| `directory` | `str` | `"./"` | Target filesystem directory where checkpoints will be stored and discovered. |
| `batch_size` | `int` | `50` | Number of records to process before computing telemetry and saving a checkpoint. |
| `start_index` | `int | None` | `None` | Explicit row index to start/resume from. If `None`, auto-detects from disk checkpoints. |
| `website_col` | `str` | `"investee_website"` | Name of the DataFrame column containing website domains or URLs. |
| `date_col` | `str` | `"investment_date"` | Name of the DataFrame column containing event or investment dates. |
| `before_months` | `int` | `0` | Months prior to the target date to include in the CDX search interval. |
| `after_months` | `int` | `18` | Months following the target date to include in the CDX search interval. |
| `max_sample_time` | `float` | `20.0` | Maximum allowed runtime (in seconds) for scraping a single row before timing out. |
| `scrape_subpages` | `bool` | `True` | Whether to discover and scrape internal subpages linked from the landing page. |
| `max_subpages` | `int` | `5` | Maximum number of internal subpages to scrape per website. |
| `delay_range` | `tuple` | `(1.0, 2.0)` | `(min, max)` uniform random sleep interval (seconds) between interval searches. |
| `timeout` | `float` | `6.0` | Network request timeout (in seconds) for CDX queries and HTML fetches. |
| `file_format` | `str` | `"parquet"` | Storage format for checkpoints: `"parquet"` or `"pickle"`. |

---

## Outputs

### 1. In-Memory Return

Returns the fully populated `pd.DataFrame` containing all original columns along with flattened scraping columns:

* **Landing Page Attributes:**
* `snapshot_date`: Capture date (`YYYY-MM-DD`) of the archive.
* `snapshot_error`: Recorded error message, timeout notice, or `None` on success.
* `home_url`: Full Wayback Machine snapshot URL.
* `home_title`: Extracted page title.
* `home_text`: Trafilatura-extracted main body text formatted as Markdown.
* `home_word_count`: Extracted alphanumeric word count.
* `home_raw_html`: Gzip-compressed, Base64-encoded raw HTML string.


* **Subpage Attributes ($i = 1$ to `max_subpages`):**
* `subpage{i}_url`: Wayback snapshot URL of the subpage.
* `subpage{i}_title`: Title of the subpage.
* `subpage{i}_text`: Clean text of the subpage.
* `subpage{i}_word_count`: Alphanumeric word count of the subpage text.
* `subpage{i}_raw_html`: Gzip-compressed, Base64-encoded HTML of the subpage.



### 2. Disk Checkpoints

Saves incremental files to `directory` named:

```text
<clean_out_name>_<processed_count>.<file_format>

```

*(e.g., `./scraped_data_50.parquet`, `./scraped_data_100.parquet`)*

