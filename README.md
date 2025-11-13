# data_enrichment
My VS Code  for the data enrichment python code

"""

People Enricher v1 — Email‑first company crawler with person extraction, icebreaker, and Excel output



What this tool does (in short)
- Reads a list of websites from an Excel workbook and processes them one by one
- Renders pages with Playwright (Chromium) and discovers “person” pages (Impressum, Team, About, etc.)
- Extracts emails (text + mailto:) first, with optional strict same‑domain filtering
- If NO email is found: writes a company‑only row and moves to the next company (skips LLM)
- If emails ARE found: calls an LLM on a small number of pages to extract people (name, role, gender, salutation)
- Generates a short, German icebreaker sentence from the homepage text (plain text, no JSON)
- Writes rows to an “Enriched Sheet” using header‑aware appends (always to the next real free row)


Key capabilities
- Robust email extraction and de‑obfuscation from rendered HTML and mailto links
- Optional strict domain matching for emails (same company domain)
- LLM‑based person extraction with deterministic salutation rules:
  - female + last_name → “Sehr geehrte Frau [last_name]”
  - male + last_name → “Sehr geehrter Herr [last_name]”
  - female + no last_name → “Liebe [first_name]”
  - male + no last_name → “Lieber [first_name]”
  - otherwise → empty string
- Icebreaker in natural German from homepage only (no JSON), sanitized to avoid trivial outputs
- Excel lock detection (stops if workbook is open/locked) and header‑aware writing (appends directly after last real row)


Where your data comes from
- Analyse sheet (input):
  - Column A: Company Name
  - Column B: Website URL
- Enriched Sheet (output): header‑aware, no hardcoded column indices; keys match header text exactly
  - Company Name, Website URL, Company summary, Job title, First name, Last name,
    Gender, Salutation, Emails, Icebreaker sentence

    
Processing flow (step by step)
1) Pre‑flight
   - Verifies the Excel file exists and is not open/locked (fails fast with a clear message if locked)
   - Loads the input workbook and determines which websites still need processing (skips already enriched ones by host)
2) For each website
   a) Render homepage with Playwright (Chromium) and extract:
      - Text emails (de‑obfuscated) + mailto emails → filtered by domain rule if enabled
      - Discover person‑relevant links (Impressum/Team/About/etc.)
   b) Phase 1 (Email‑first): pre‑scan the discovered pages for emails ONLY (no LLM yet)
      - If no emails are found at all → write a company‑only row and go to the next company
   c) Company summary + icebreaker
      - Summarize homepage into a short German sentence (company_summary)
      - Generate a plain‑text German “icebreaker” (no JSON), sanitized; may be empty if nothing concrete
   d) Phase 2 (LLM people extraction, limited pages)
      - For up to LLM_MAX_PAGES pages, call the LLM to extract people (first/last name, role, gender, salutation)
      - Person emails are accepted only if they were actually parsed from the site (no made‑up emails)
      - If exactly one global email exists and a person has no email, assign that one
      - If emails exist but no people → write company‑only row with generic salutation (“Sehr geehrte Damen und Herren”) and all emails
   e) Excel write (header‑aware)
      - Appends rows directly after the last populated row (ignores Excel’s stale used‑range)
      - Prints true before/after row numbers (based on key columns), not ws.max_row

Configuration (edit these constants near the top of the file)
- EXCEL_PATH: absolute path to your workbook (xlsx)
- INPUT_SHEET_NAME: input sheet name (e.g., "Analyse")
- OUTPUT_SHEET_NAME: output sheet name (e.g., "Enriched Sheet"); must exist with expected headers
- HEADLESS: True to run Chromium without UI
- IGNORE_ROBOTS: True during testing; set False to respect robots.txt
- NAV_TIMEOUT_MS, PAGE_TIMEOUT_S, DOMAIN_TIMEOUT_S: timeouts for navigation/page load and per‑site watchdog
- STRICT_DOMAIN_ONLY: False by default; set True to accept only emails from the same company domain
- MAX_PERSON_LINKS: cap how many person‑relevant links to scan per site
- VERBOSE: True shows detailed logs; set False later for quiet runs
- MAX_ROWS_TO_PROCESS: limit how many new rows to process this run (for testing)
- POST_COMPANY_SLEEP_S: wait time between companies (to keep within LLM rate limits)
- LLM_MAX_PAGES: max number of pages to send to LLM per site (0 = unlimited)

LLM configuration
- MODEL_NAME: set via environment (OPENAI_MODEL), default "gpt-4o-mini". For higher quality, try "gpt-4o"
- MAX_LLM_CALLS_PER_MIN: coarse client‑side throttle to avoid rate errors
- MAX_PAGE_TEXT_CHARS: truncate per‑page text sent to LLM
- OPENAI_API_KEY must be set via Key.env/.env or environment variable

Run requirements
- Python packages: playwright, bs4, tldextract, openpyxl, openai, tenacity, python-dotenv (optional)
  Install:
    pip install playwright bs4 tldextract openpyxl openai tenacity python-dotenv
    python -m playwright install chromium
- Environment key: OPENAI_API_KEY (via Key.env/.env or export)

How to run
- Put websites in Analyse (A: Company Name, B: Website URL)
- Close Excel before running (the script checks and aborts if the file is open)
- Run:
    python3 people_enricher_v1.py
- Watch the console for true before/after row counts and per‑page logs

Troubleshooting
- "File appears open/locked": Close the workbook in Excel/Numbers and re‑run
- Icebreaker is empty: It’s allowed if the homepage contains nothing concrete; try a better model (e.g., gpt-4o)
- No rows written: Verify sheet names/headers and paths, and that the site has any emails

Compliance
- Emails and possible personal data are scraped from public pages. Ensure your usage complies with applicable laws (e.g., GDPR) and website terms.
"""
