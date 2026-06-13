#!/usr/bin/env python3
"""
FORGE (Funding Opportunity Reconnaissance & Grant Engine)
An AI-backed grant discovery and ranking tool.
"""

######################################
# Metadata
######################################
__authors__ = "Troy Baker, Andrew Kapaldo, Barry Thomas, Logan Vance, & Sophia Walker"
__version__ = "0.4.0"
__status__ = "Development"

######################################
# Imports
######################################
import os
import sys
import time
import csv
import json
import logging
import argparse
from datetime import datetime
import base64

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry
import instructor
from openai import OpenAI, RateLimitError
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

######################################
# Global Application Settings
######################################

# --- Grants.gov Search Settings ---
GRANTS_API_URL = "https://api.grants.gov/v1/api/search2"
GRANTS_FETCH_URL = "https://api.grants.gov/v1/api/fetchOpportunity"
# Consider moving these to an external config or .env in the future
SEARCH_KEYWORDS = ["STEM", "highschool", "high school", "makerspace", "robotics"]
SEARCH_ROWS_PER_KEYWORD = 10
SEARCH_STATUSES = "posted|forecasted"
SEARCH_ELIGIBILITY = "12"

# --- AI Engine Settings ---
AI_BASE_URL = "https://api.groq.com/openai/v1"
AI_MODEL_NAME = "llama-3.1-8b-instant"
AI_TEMPERATURE = 0.0
AI_SEED = 42
MAX_AI_RETRIES = 3
RATE_LIMIT_WAIT_SECONDS = 15
DEEP_DIVE_THRESHOLD = 70
AI_CALL_DELAY_SECONDS = 2
HTTP_MAX_RETRIES = 3
HTTP_BACKOFF_FACTOR = 1
HTTP_RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# --- File Configuration ---
PROFILE_FILE = "client_profile.txt"
FEEDBACK_FILE = "feedback_log.csv"
RAW_DATA_FILE = "grants_output.csv"
HTML_TEMPLATE_FILE = "report_template.html"
FINAL_REVIEW_FILE = "daily_grants_review.html"
LOG_FILE = "forge.log"

######################################
# Logging & Terminal Setup
######################################
class _StripAnsiFormatter(logging.Formatter):
    """Strips ANSI color codes from log records written to file."""
    import re
    _ANSI_RE = re.compile(r'\033\[[0-9;]*m')

    def format(self, record):
        record.msg = self._ANSI_RE.sub('', str(record.msg))
        return super().format(record)

def setup_logging(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("FORGE")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = "%(asctime)s [%(levelname)-8s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(_StripAnsiFormatter(fmt=fmt, datefmt=datefmt))
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class bcolors:
    _tty = sys.stdout.isatty()
    HEADER  = '\033[91m' if _tty else ''
    BLUE    = '\033[94m' if _tty else ''
    CYAN    = '\033[96m' if _tty else ''
    GREEN   = '\033[92m' if _tty else ''
    WARNING = '\033[93m' if _tty else ''
    FAIL    = '\033[91m' if _tty else ''
    PURPLE  = '\033[95m' if _tty else ''
    END     = '\033[0m'  if _tty else ''
    BOLD    = '\033[1m'  if _tty else ''

######################################
# Output Schema for AI
######################################
class GrantEvaluation(BaseModel):
    alignment_score: int = Field(
        ...,
        description=(
            "Score 0-100 indicating mission alignment. "
            "Use highly specific, granular numbers. "
            "DO NOT simply round to the nearest 10."
        )
    )
    justification: str = Field(
        ...,
        description="Strict 2-sentence explanation of the score."
    )
    is_eligible: bool = Field(
        ...,
        description=(
            "True if eligible, False if there are red flags. "
            "MUST be a valid JSON boolean (true or false)."
        )
    )

######################################
# Core Initialization
######################################
def build_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=HTTP_MAX_RETRIES,
        backoff_factor=HTTP_BACKOFF_FACTOR,
        status_forcelist=HTTP_RETRY_STATUS_CODES,
        allowed_methods=["POST", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

load_dotenv()
api_key = os.getenv("AI_API_KEY")
if not api_key:
    print(f"{bcolors.FAIL}[!] ERROR: AI_API_KEY environment variable is missing.{bcolors.END}")
    sys.exit(1)

ai_client = instructor.from_openai(
    OpenAI(base_url=AI_BASE_URL, api_key=api_key),
    mode=instructor.Mode.JSON
)

######################################
# Helper & I/O Functions
######################################
def load_client_profile(filepath: str = PROFILE_FILE) -> str:
    try:
        with open(filepath, mode='r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"{bcolors.FAIL}[!] ERROR: Could not find '{filepath}'.{bcolors.END}")
        sys.exit(1)

def build_system_prompt(profile_text: str, feedback_file: str = FEEDBACK_FILE) -> str:
    prompt = f"""
    You are an expert grant reviewer for the following non-profit:
    {profile_text}

    Evaluate incoming grants based ONLY on the provided information. Be highly critical.
    If the grant is irrelevant to the mission, give it a low score.
    """
    if os.path.exists(feedback_file):
        prompt += "\n\n### PAST HUMAN FEEDBACK (LEARN FROM THESE MISTAKES) ###\n"
        with open(feedback_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt += f"- Grant: '{row.get('Title', 'Unknown')}'\n"
                prompt += (
                    f"  * Human Correction: Score should be {row.get('Human_Score', '0')} "
                    f"because: {row.get('Human_Notes', '')}\n"
                )
    return prompt

def save_raw_grants_to_csv(grants_list: list[dict], filepath: str = RAW_DATA_FILE):
    """Saves raw grant data to disk purely for backup/audit purposes."""
    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Opportunity Number", "Title", "Agency", "Status", "Open Date", "Close Date", "Internal ID", "Search Keyword"])
        for grant in grants_list:
            writer.writerow([
                grant.get("number", "N/A"),
                grant.get("title",  "No Title"),
                grant.get("agency", "Unknown"),
                grant.get("oppStatus", "Unknown"),
                grant.get("openDate", "N/A"),
                grant.get("closeDate", "N/A"),
                grant.get("id", ""),
                grant.get("forge_keyword", "Unknown")
            ])

######################################
# API Interaction Modules
######################################
def fetch_single_grant(opportunity_number: str, http: requests.Session, log: logging.Logger) -> dict | None:
    """Fetches base metadata for a single manually entered grant."""
    payload = {
        "keyword": opportunity_number,
        "oppStatuses": "posted|forecasted|closed|archived"
    }
    try:
        response = http.post(GRANTS_API_URL, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
        response.raise_for_status()
        hits = response.json().get('data', {}).get('oppHits', [])
        
        # Ensure exact match
        for hit in hits:
            if hit.get('number') == opportunity_number:
                hit['forge_keyword'] = "Manual Entry"
                return hit
                
        log.warning("Could not find an exact metadata match for manual grant '%s'.", opportunity_number)
        return None
    except RequestException as e:
        log.error("Error fetching manual grant '%s': %s", opportunity_number, e)
        return None

def fetch_grants_by_keywords(http: requests.Session, log: logging.Logger) -> list[dict]:
    """Phase 1: Harvest grants from Grants.gov based on standard keywords."""
    unique_grants: dict[str, dict] = {}
    for kw in SEARCH_KEYWORDS:
        payload = {
            "rows": SEARCH_ROWS_PER_KEYWORD,
            "oppStatuses": SEARCH_STATUSES,
            "eligibilities": SEARCH_ELIGIBILITY,
            "keyword": kw,
        }
        try:
            response = http.post(GRANTS_API_URL, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
            response.raise_for_status()
            api_data  = response.json().get('data', {})
            opp_hits  = api_data.get('oppHits', [])
            
            log.info("%s↳ '%s': Found %d total grants (pulling top %d)%s", bcolors.BLUE, kw, api_data.get('hitCount', 0), len(opp_hits), bcolors.END)

            for grant in opp_hits:
                if grant.get("number") and grant["number"] not in unique_grants:
                    grant["forge_keyword"] = kw
                    unique_grants[grant["number"]] = grant
            time.sleep(0.5)
        except RequestException as e:
            log.error("%s[-] Error fetching keyword '%s': %s%s", bcolors.FAIL, kw, e, bcolors.END)
            
    return list(unique_grants.values())

def fetch_grant_synopsis(opportunity_number: str, internal_id: str, http: requests.Session, log: logging.Logger) -> str | None:
    """Fetches the deep-dive long-form text. Optimized to skip the search if internal_id is already known."""
    try:
        # If we didn't capture the internal ID previously, we have to look it up
        if not internal_id:
            matched_grant = fetch_single_grant(opportunity_number, http, log)
            if not matched_grant or not matched_grant.get('id'):
                return None
            internal_id = matched_grant.get('id')

        details_response = http.post(
            GRANTS_FETCH_URL,
            headers={"Content-Type": "application/json"},
            json={"opportunityId": internal_id},
            timeout=30,
        )
        details_response.raise_for_status()

        full_data = details_response.json().get('data', {})
        synopsis_dict = full_data.get('synopsis', {})
        detailed_text = synopsis_dict.get('synopsisDesc') or full_data.get('description')

        if detailed_text:
            clean_text = BeautifulSoup(detailed_text, "html.parser").get_text(separator="\n", strip=True)
            return clean_text if clean_text.strip() else None

        return json.dumps(full_data)
    except RequestException as e:
        log.warning("Failed to fetch Deep Dive synopsis for '%s': %s", opportunity_number, e)
        return None

######################################
# AI Processing Modules
######################################
def evaluate_grant_with_ai(grant_title: str, grant_agency: str, system_prompt: str, log: logging.Logger, synopsis: str | None = None) -> GrantEvaluation:
    user_prompt = f"Agency: {grant_agency} | Title: {grant_title}"
    if synopsis:
        user_prompt += f"\n\n### FULL GRANT SYNOPSIS DETAILS ###\n{synopsis}"

    for attempt in range(MAX_AI_RETRIES):
        try:
            return ai_client.chat.completions.create(
                model=AI_MODEL_NAME,
                response_model=GrantEvaluation,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=AI_TEMPERATURE,
                seed=AI_SEED,
                stream=False,
            )
        except RateLimitError:
            if attempt < MAX_AI_RETRIES - 1:
                log.warning("AI rate limit hit (attempt %d/%d). Waiting %ds...", attempt + 1, MAX_AI_RETRIES, RATE_LIMIT_WAIT_SECONDS)
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
            else:
                raise

def process_and_score_grants(grants_to_process: list[dict], master_prompt: str, http: requests.Session, log: logging.Logger) -> list[dict]:
    """Iterates through the in-memory list of grants, scores them, and runs deep dives."""
    evaluated_grants_list = []
    total_grants = len(grants_to_process)

    for row_count, grant in enumerate(grants_to_process, start=1):
        grant_number   = grant.get("number", "N/A")
        grant_title    = grant.get("title", "No Title")
        grant_agency   = grant.get("agency", "Unknown")
        internal_id    = grant.get("id", "")
        search_keyword = grant.get("forge_keyword", "Unknown")
        
        grant_url = f"https://www.grants.gov/search-results-detail/{internal_id}" if internal_id else f"https://www.grants.gov/search?keyword={grant_number}"

        log.info("%s[%d/%d] Triage Pass:%s %s...", bcolors.BOLD, row_count, total_grants, bcolors.END, grant_title[:60])

        try:
            # --- Pass 1: Quick Check ---
            ai_result     = evaluate_grant_with_ai(grant_title, grant_agency, master_prompt, log)
            initial_score = ai_result.alignment_score
            deep_dive     = False

            log.info("    ↳ Initial Score: %d/100", initial_score)
            time.sleep(AI_CALL_DELAY_SECONDS)

            # --- Pass 2: Deep Dive ---
            if initial_score > DEEP_DIVE_THRESHOLD:
                log.info("    %s↳ Score > %d! Initiating Deep Dive...%s", bcolors.PURPLE, DEEP_DIVE_THRESHOLD, bcolors.END)
                synopsis = fetch_grant_synopsis(grant_number, internal_id, http, log)

                if synopsis:
                    ai_result = evaluate_grant_with_ai(grant_title, grant_agency, master_prompt, log, synopsis=synopsis)
                    deep_dive = True
                    time.sleep(AI_CALL_DELAY_SECONDS)
                else:
                    log.warning("    ↳ Deep Dive skipped for '%s' (No data).", grant_number)

            score_color = bcolors.GREEN if ai_result.alignment_score >= 80 else bcolors.WARNING if ai_result.alignment_score >= 60 else bcolors.FAIL
            log.info("    ↳ %sFinal Verdict: %d/100 | Eligible: %s | Deep Dive: %s%s", score_color, ai_result.alignment_score, ai_result.is_eligible, deep_dive, bcolors.END)

            evaluated_grants_list.append({
                "title": grant_title,
                "agency": grant_agency,
                "number": grant_number,
                "url": grant_url,
                "keyword": search_keyword,
                "score": ai_result.alignment_score,
                "eligible": ai_result.is_eligible,
                "justification": ai_result.justification,
                "deep_dive": deep_dive
            })

        except Exception as e:
            log.error("    ↳ ERROR processing '%s': %s", grant_title[:60], e)

    return evaluated_grants_list

######################################
# Output & Reporting Modules
######################################
def generate_html_report(evaluated_grants_list: list[dict], log: logging.Logger):
    """Renders the final Jinja2 HTML report."""
    try:
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template(HTML_TEMPLATE_FILE)
    except Exception as e:
        log.critical("%s[!] Cannot load HTML template '%s': %s%s", bcolors.FAIL, HTML_TEMPLATE_FILE, e, bcolors.END)
        sys.exit(1)

    try:
        with open("./images/forge_logo.png", "rb") as img_file:
            logo_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        log.warning("Logo image not found. Proceeding without it.")
        logo_base64 = ""

    html_output = template.render(
        grants=sorted(evaluated_grants_list, key=lambda x: x['score'], reverse=True),
        run_date=datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        logo_data=logo_base64
    )
        
    with open(FINAL_REVIEW_FILE, 'w', encoding='utf-8') as f:
        f.write(html_output)

######################################
# Main Pipeline Orchestrator
######################################
def main():
    parser = argparse.ArgumentParser(description="FORGE: Funding Opportunity Reconnaissance & Grant Engine")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-m", "--manual", type=str, help="Manually evaluate a specific Grants.gov Opportunity Number (e.g., HRSA-24-001)")
    args = parser.parse_args()

    log = setup_logging(verbose=args.verbose)
    http = build_http_session()

    print(f'''{bcolors.HEADER}{bcolors.BOLD}
 ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
 ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
 █████╗  ██║   ██║██████╔╝██║  ███╗█████╗
 ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
 ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
 ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
 {bcolors.END}''')
    log.info("FORGE v%s | Authors: %s", __version__, __authors__)

    # ---------------------------------------------------------
    # PHASE 1: DATA INGESTION
    # ---------------------------------------------------------
    grants_to_process = []
    
    if args.manual:
        log.info("%sPHASE 1: Manual Input Mode Triggered for '%s'%s", bcolors.CYAN, args.manual, bcolors.END)
        manual_grant = fetch_single_grant(args.manual, http, log)
        if manual_grant:
            grants_to_process.append(manual_grant)
            log.info("%s[+] Successfully loaded metadata for Manual Entry.%s", bcolors.GREEN, bcolors.END)
        else:
            log.critical("%s[!] Could not locate grant metadata on Grants.gov. Exiting.%s", bcolors.FAIL, bcolors.END)
            sys.exit(1)
    else:
        log.info("%sPHASE 1: Fetching data from Grants.gov...%s", bcolors.CYAN, bcolors.END)
        grants_to_process = fetch_grants_by_keywords(http, log)
        log.info("%s[+] API Query Complete! %d unique grants forwarded to AI Engine.%s", bcolors.GREEN, len(grants_to_process), bcolors.END)

    # Save to disk just for backup/auditing (no longer required to proceed)
    save_raw_grants_to_csv(grants_to_process)

    # ---------------------------------------------------------
    # PHASE 2: AI SCORING
    # ---------------------------------------------------------
    log.info("%sPHASE 2: AI Triage & Deep Dive Engine...%s", bcolors.CYAN, bcolors.END)
    profile = load_client_profile()
    master_prompt = build_system_prompt(profile)

    evaluated_grants_list = process_and_score_grants(grants_to_process, master_prompt, http, log)
    
    log.info("%s[+] Finished processing %d grants. Generating reports...%s", bcolors.CYAN, len(evaluated_grants_list), bcolors.END)

    # ---------------------------------------------------------
    # PHASE 3: REPORTING
    # ---------------------------------------------------------
    with open("grants_cache.json", "w", encoding="utf-8") as cache_file:
        json.dump(evaluated_grants_list, cache_file, indent=4)

    generate_html_report(evaluated_grants_list, log)
    
    log.info("%s%s[+] FORGE Run Complete! Final HTML report saved to '%s'.%s", bcolors.GREEN, bcolors.BOLD, FINAL_REVIEW_FILE, bcolors.END)

if __name__ == "__main__":
    main()
