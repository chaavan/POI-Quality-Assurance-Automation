# projectd.py
# -*- coding: utf-8 -*-

import time
import requests
from bs4 import BeautifulSoup
from typing import Any, List, Dict
import concurrent.futures
import dspy
from pydantic import BaseModel

from overturemaps import core

# ─── DSPY & LLM SETUP ─────────────────────────────────────────────────────────

# Example: Gemini; you can swap this out for any other provider
lm = dspy.LM(
    'gemini/gemini-2.0-flash', 
    api_key='AIzaSyC_YvqPom_7xgJCt2V6QLSH8LQpsbH9dzM'
)
dspy.configure(lm=lm)

# ─── UTILS ───────────────────────────────────────────────────────────────────

def scrape_website(url: str, timeout: float = 5.0) -> str:
    """Fetches title, meta-description, and H1s from a URL, returns as single string."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        title = soup.title.string if soup.title else 'No title'
        desc_tag = soup.find('meta', {'name': 'description'})
        desc = desc_tag['content'] if desc_tag and 'content' in desc_tag.attrs else ''

        h1s = '; '.join(h.get_text(strip=True) for h in soup.find_all('h1'))
        return f"Title: {title}. Description: {desc}. H1s: {h1s}"
    except Exception as e:
        print(f"[WARN] scrape_website({url}) failed:", e)
        return ""

def scrape_place_website(place_name: str, websites: dict) -> str:
    """Scrape the first URL in the `websites` list, if any."""
    if not websites:
        return ""
    first = websites[0]
    if not first:
        return ""
    return scrape_website(first)

# ─── DSPY SIGNATURE & MODULE ─────────────────────────────────────────────────

# Category Validation
class CategoryValidation(dspy.Signature):
    """Is the current category correct? If not, suggest one."""
    poi_name: str = dspy.InputField()
    poi_description: str = dspy.InputField()
    current_category: str = dspy.InputField()
    response: str = dspy.OutputField()

class CategoryValidator(dspy.Module):
    def __init__(self):
        self.query = dspy.Predict(CategoryValidation)

    def forward(
            self, 
            poi_name: str, 
            poi_description: str, 
            current_category: str) -> str:
        out = self.query(
            poi_name=poi_name,
            poi_description=poi_description,
            current_category=current_category
        )
        return out.response

# Completeness Check
class CompletenessCheck(dspy.Signature):
    """Check which required fields are missing from a POI record."""
    record: Dict[str, Any]     = dspy.InputField()
    required_fields: List[str] = dspy.InputField()
    missing_fields: List[str]  = dspy.OutputField()
    # allow arbitrary nested types (dicts, lists, etc.)
    model_config = {"arbitrary_types_allowed": True}

class CompletenessChecker(dspy.Module):
    def __init__(self):
        self.query = dspy.Predict(CompletenessCheck)

    def forward(self, record: dict, required_fields: List[str]) -> List[str]:
        out = self.query(
            record=record, 
            required_fields=required_fields
            )
        return out.missing_fields

# Address Formatting
class AddressFormat(dspy.Signature):
    address: str = dspy.InputField()
    corrected: str = dspy.OutputField()

class AddressFormatter(dspy.Module):
    def __init__(self):
        self.query = dspy.Predict(AddressFormat)

    def forward(self, address: str) -> str:
        out = self.query(
            address=address
            )
        return out.corrected

# Duplicate Detection
class DuplicateDetection(dspy.Signature):
    """Decide if two POI records refer to the same real-world place."""
    rec1: Dict[str, Any] = dspy.InputField()
    rec2: Dict[str, Any] = dspy.InputField()
    is_duplicate: bool = dspy.OutputField()
    reason: str = dspy.OutputField()
    # allow arbitrary nested types
    model_config = {"arbitrary_types_allowed": True}

class DuplicateDetector(dspy.Module):
    def __init__(self):
        self.query = dspy.Predict(DuplicateDetection)

    def forward(self, rec1: dict, rec2: dict) -> Dict[str, Any]:
        out = self.query(
            rec1=rec1, 
            rec2=rec2
            )
        return {"duplicate": out.is_duplicate, "reason": out.reason}
# ─── PIPELINE & ORCHESTRATION ────────────────────────────────────────────────

class POIQAPipeline:
    """Orchestrates scraping + category validation (extendable!)."""

    def __init__(
            self, 
            validator: CategoryValidator,
            completeness: CompletenessChecker,
            formatter: AddressFormatter,
            duplicate_detector: DuplicateDetector,
            websites: dict
            ):
        self.validator = validator
        self.completeness = completeness
        self.formatter = formatter
        self.duplicate_detector = duplicate_detector
        self.websites = websites

        self.call_count = 0  # for rate-limiting
        self.rate_limit = 15 
    
    def _api_call(self, fn, *args):
        """Wrapper: call the module fn, increment counter, pause every 10th call."""
        result = fn(*args)
        self.call_count += 1
        if self.call_count % self.rate_limit == 0:
            print(f"[RATE LIMIT] {self.call_count} LLM calls made—sleeping 60 s…")
            time.sleep(60)
        return result

    def validate_one(self, record: Dict[str, Any]) -> Dict[str, Any]:
        name       = record.get("name", "")
        category   = record.get("category", "")
        urls       = record.get("websites", [])
        socials    = record.get("socials", [])
        emails     = record.get("emails", [])
        phones     = record.get("phones", [])
        addresses  = record.get("addresses", [])

        # 1. Scrape description from website
        desc = scrape_place_website(name, urls)

        # 2. Category suggestion
        cat_suggestion = self._api_call(self.validator, name, desc, category)

        # 3. Completeness (now checks all key props)
        required = ["name","category","websites","socials","emails","phones","addresses"]
        missing = self._api_call(self.completeness, record, required)

        # 4. Address formatting (use the first listed address if any)
        first_addr = addresses[0] if addresses else ""
        corrected_address = self._api_call(self.formatter, first_addr) if first_addr else ""

        return {
            "name": name,
            "category_suggestion": cat_suggestion,
            "missing_fields": missing,
            "corrected_address": corrected_address
        }
    
    def detect_duplicates(self, places: List[dict]) -> List[tuple]:
        dups = []
        for i, rec1 in enumerate(places):
            for rec2 in places[i+1:]:
                res = self.duplicate_detector(rec1, rec2)
                if res["duplicate"]:
                    dups.append((rec1["name"], rec2["name"], res["reason"]))
        return dups
    
    def run(self, places: List[dict]) -> Dict[str, Any]:
        qa_results = []
        for i, rec in enumerate(places):
            print(f"Validating {i+1}/{len(places)}: {rec['name']}")
            qa_results.append(self.validate_one(rec))

        duplicates = self.detect_duplicates(places)
        return {"qa": qa_results, "duplicates": duplicates}

# ─── MAIN EXECUTION ───────────────────────────────────────────────────────────

def main():
    # 1. Load Overture Places
    bbox = (-117.003, 33.05, -117, 33.1)
    gdf  = core.geodataframe("place", bbox=bbox)
    print("Loaded", gdf.shape[0], "places")

    # 2. Build helper maps & records
    websites = {
        gdf.names[i]['primary']: gdf.websites[i]
        for i in range(len(gdf))
    }
    places = []
    for i in range(len(gdf)):
        name = gdf.names[i]['primary']
        places.append({
            "name": name,
            "category": gdf.categories[i]["primary"],
            "confidence": gdf.confidence[i],
            "websites": gdf.websites[i],
            "socials": gdf.socials[i],
            "emails": gdf.emails[i],
            "phones": gdf.phones[i],
            "brand": gdf.brand[i],
            "addresses": gdf.addresses[i]
        })


    # 3. Instantiate pipeline
    pipeline = POIQAPipeline(
        validator=CategoryValidator(),
        completeness=CompletenessChecker(),
        formatter=AddressFormatter(),
        duplicate_detector=DuplicateDetector(),
        websites=websites
    )

    # 4. Run it
    output = pipeline.run(places)

    # 5. Print / store results
    print("\n--- QA RESULTS ---")
    for rec in output["qa"]:
        print(rec)

    if output["duplicates"]:
        print("\n--- POTENTIAL DUPLICATES ---")
        for a, b, reason in output["duplicates"]:
            print(f"  • {a} ↔ {b}: {reason}")

if __name__ == "__main__":
    main()
