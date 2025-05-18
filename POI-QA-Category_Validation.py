# projectd.py
# -*- coding: utf-8 -*-

import streamlit as st
import json
import time
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import pandas as pd

import dspy
from overturemaps import core

# ─── DSPY & LLM SETUP ─────────────────────────────────────────────────────────
# lm will be configured in the Streamlit app based on user input
lm = None

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
        # This warning will go to the console where Streamlit is running
        print(f"[WARN] scrape_website({url}) failed:", e)
        return ""

def scrape_place_website(place_name: str, urls: List[str]) -> str:
    """Scrape the first URL in the `urls` list, if any."""
    if not urls:
        return ""
    first_url = urls[0]
    if not first_url or not isinstance(first_url, str): # Ensure it's a non-empty string
        return ""
    return scrape_website(first_url)

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
        super().__init__()
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
    model_config = {"arbitrary_types_allowed": True}

class CompletenessChecker(dspy.Module):
    def __init__(self):
        super().__init__()
        self.query = dspy.Predict(CompletenessCheck)

    def forward(self, record: dict, required_fields: List[str]) -> List[str]:
        out = self.query(
            record=record, 
            required_fields=required_fields
            )
        return out.missing_fields

# Address Formatting
class AddressFormat(dspy.Signature):
    """Given an address string, correct its formatting if necessary or return the original if correct."""
    address: str = dspy.InputField()
    corrected: str = dspy.OutputField()

class AddressFormatter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.query = dspy.Predict(AddressFormat)

    def forward(self, address: str) -> str:
        out = self.query(
            address=address
            )
        return out.corrected

# Duplicate Detection
class DuplicateDetection(dspy.Signature):
    """Decide if two POI records refer to the same real-world place."""
    rec1: Dict[str, str] = dspy.InputField()
    rec2: Dict[str, str] = dspy.InputField()
    is_duplicate: bool = dspy.OutputField()
    reason: str = dspy.OutputField()
    model_config = {"arbitrary_types_allowed": True}

class DuplicateDetector(dspy.Module):
    def __init__(self):
        super().__init__()
        self.query = dspy.Predict(DuplicateDetection)

    def forward(self, rec1: dict, rec2: dict) -> Dict[str, Any]:
        out = self.query(
            rec1=rec1, 
            rec2=rec2
            )
        return {"duplicate": out.is_duplicate, "reason": out.reason}

# ─── PIPELINE & ORCHESTRATION ────────────────────────────────────────────────

class POIQAPipeline:
    """Orchestrates scraping + QA modules based on selection."""

    def __init__(
            self, 
            validator: CategoryValidator,
            completeness: CompletenessChecker,
            formatter: AddressFormatter,
            duplicate_detector: DuplicateDetector
            ):
        self.validator = validator
        self.completeness = completeness
        self.formatter = formatter
        self.duplicate_detector = duplicate_detector

        self.call_count = 0
        self.rate_limit = 15 # Number of LLM calls before pausing
    
    def _api_call(self, fn, *args):
        """Wrapper: call the module fn, increment counter, pause if rate limit hit."""
        if not dspy.settings.lm:
            st.error("LLM not configured. Cannot make API call.")
            raise Exception("LLM not configured for API call.")
        
        result = fn(*args)
        self.call_count += 1
        if self.call_count % self.rate_limit == 0:
            # This message goes to the console where Streamlit server is running.
            # The Streamlit app UI will appear to hang during the sleep.
            pause_msg = st.warning(f"⚠️ Rate limit reached ({self.call_count} calls). Pausing 60s …")
            time.sleep(60)
            pause_msg.empty()
        return result

    def validate_one(self, record: Dict[str, Any], selected_qa: List[str]) -> Dict[str, Any]:
        name       = record.get("name", "")
        category   = record.get("category", "")
        urls       = record.get("websites", []) # This is List[str]
        addresses  = record.get("addresses", [])

        # Initialize result with name, and default "Not performed" for all QA fields
        result = {
            "name": name,
            "category_suggestion": "Not performed",
            "missing_fields": "Not performed",
            "corrected_address": "Not performed"
        }

        # 1. Scrape description (only if Category Validation is selected)
        desc = ""
        if "Category Validation" in selected_qa:
            desc = scrape_place_website(name, urls)
            cat_suggestion = self._api_call(self.validator.forward, name, desc, category)
            result["category_suggestion"] = cat_suggestion
        
        # 2. Completeness Check
        if "Completeness Check" in selected_qa:
            required = ["name","category","websites","socials","emails","phones","addresses"]
            missing = self._api_call(self.completeness.forward, record, required)
            result["missing_fields"] = missing if missing else [] # Ensure it's a list

        # 3. Address Formatting
        if "Address Formatting" in selected_qa:
            first_addr_dict = addresses[0] if addresses and isinstance(addresses[0], dict) else {}
            first_addr_str = first_addr_dict.get("freeform", "") if first_addr_dict else ""
            
            if first_addr_str:
                corrected_address = self._api_call(self.formatter.forward, first_addr_str)
                result["corrected_address"] = corrected_address
            else:
                result["corrected_address"] = "No address to format"
        
        return result
    
    def detect_duplicates(self, places: List[dict]) -> List[tuple]:
        dups = []
        # For duplicate detection, we need string representations of records.
        # Let's simplify records to include only key textual fields for comparison.
        def simplify_record(rec):
            return {
                "name": str(rec.get("name", "")),
                "category": str(rec.get("category", "")),
                "address": str(rec.get("addresses")[0].get("freeform", "") if rec.get("addresses") else ""),
                "website": str(rec.get("websites")[0] if rec.get("websites") else "")
            }

        simplified_places = [simplify_record(p) for p in places]

        for i, rec1_simple in enumerate(simplified_places):
            for j, rec2_simple in enumerate(simplified_places[i+1:]):
                # Pass original names for reporting, but simplified records for detection
                res =self.duplicate_detector.forward (rec1_simple, rec2_simple)
                if res.get("duplicate"): # Check if 'duplicate' key exists and is True
                    dups.append((places[i]["name"], places[i+1+j]["name"], res.get("reason", "No reason provided")))
        return dups
    
    def run(self, places: List[dict], selected_qa: List[str]) -> Dict[str, Any]:
        qa_results = []
        
        # Perform single-record validations if any are selected
        single_record_qa_selected = any(item in selected_qa for item in ["Category Validation", "Completeness Check", "Address Formatting"])

        if single_record_qa_selected:
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, rec in enumerate(places):
                status_text.text(f"Validating POI {i+1}/{len(places)}: {rec.get('name', 'Unnamed POI')}")
                qa_results.append(self.validate_one(rec, selected_qa))
                progress_bar.progress((i + 1) / len(places))
            status_text.text(f"Validated {len(places)} POIs.")
            progress_bar.empty()

        duplicates_result = []
        if "Duplicate Detection" in selected_qa:
            with st.spinner("Detecting duplicates... This may take a while for many POIs."):
                duplicates_result = self.detect_duplicates(places)
        
        return {"qa": qa_results, "duplicates": duplicates_result}

# ─── STREAMLIT APP ───────────────────────────────────────────────────────────
def streamlit_app():
    st.set_page_config(layout="wide")
    st.title("🌍 Overture POI Quality Assurance Tool")

    st.sidebar.header("⚙️ Configuration")
    # It's better to use st.secrets for API keys in deployed apps
    api_key_input = st.sidebar.text_input(
        "Gemini API Key", 
        type="password", 
        value="", # Encourage user to input their own key
        help="Enter your Google AI Studio Gemini API Key."
    )
    
    global lm
    if api_key_input:
        try:
            lm = dspy.LM('gemini/gemini-2.0-flash', # Using a common Gemini model
                api_key=api_key_input,
                max_tokens=1000 # Example, adjust as needed
            )
            dspy.configure(lm=lm)
            st.sidebar.success("LLM Configured!")
        except Exception as e:
            st.sidebar.error(f"LLM Config Error: {e}")
            st.stop()
    else:
        st.sidebar.warning("Please enter your Gemini API Key to proceed.")
        st.stop()

    st.header("📍 Bounding Box Input")
    default_bbox = "-117.003, 33.05, -117.002, 33.051" # Smaller default for faster testing
    bbox_str = st.text_input(
        "Enter Bounding Box (min_lon, min_lat, max_lon, max_lat):", 
        default_bbox,
        help="Example: -117.005, 33.05, -117.003, 33.1"
    )

    st.header("🔍 QA Verifications")
    available_qa = [
        "Category Validation",
        "Completeness Check",
        "Address Formatting",
        "Duplicate Detection"
    ]
    selected_qa = st.multiselect(
        "Select QA Verifications to perform:", 
        available_qa, 
        default=available_qa # Select all by default
    )

    if st.button("🚀 Run QA on POIs", type="primary"):
        if not bbox_str:
            st.error("❗ Please enter a bounding box.")
            st.stop()
        if not selected_qa:
            st.error("❗ Please select at least one QA verification.")
            st.stop()
        if not dspy.settings.lm:
            st.error("❗ LLM not configured. Please check API Key in sidebar.")
            st.stop()

        try:
            bbox_parts = [float(p.strip()) for p in bbox_str.split(',')]
            if len(bbox_parts) != 4:
                raise ValueError("BBOX must have 4 parts.")
            bbox = tuple(bbox_parts)
        except ValueError as e:
            st.error(f"Invalid BBOX format: {e}. Expected 'min_lon, min_lat, max_lon, max_lat'")
            st.stop()

        with st.spinner(f"Fetching Overture Places data for BBOX: {bbox}..."):
            try:
                gdf = core.geodataframe("place", bbox=bbox)
                if gdf.empty:
                    st.warning(f"No places found for the BBOX: {bbox}. Try a different area.")
                    st.stop()
                st.info(f"🗺️ Loaded {gdf.shape[0]} places from Overture Maps.")
            except Exception as e:
                st.error(f"Error fetching Overture data: {e}")
                st.stop()
        
        places = []
        for i in range(len(gdf)):
            try:
                name = gdf.names[i]['common'][0]['value'] if gdf.names[i] and gdf.names[i].get('common') else \
                       (gdf.names[i]['primary'] if gdf.names[i] and gdf.names[i].get('primary') else f"Unnamed POI {gdf.id[i]}")
                
                record = {
                    "id": gdf.id[i],
                    "name": name,
                    "category": gdf.categories[i]["primary"] if gdf.categories[i] and "primary" in gdf.categories[i] else "N/A",
                    "confidence": gdf.confidence[i] if gdf.confidence[i] is not None else 0.0,
                    "websites": gdf.websites[i] if gdf.websites[i] else [],
                    "socials": gdf.socials[i] if gdf.socials[i] else [],
                    "emails": gdf.emails[i] if gdf.emails[i] else [],
                    "phones": gdf.phones[i] if gdf.phones[i] else [],
                    "brand": gdf.brand[i].get("names", {}).get("common", [{}])[0].get("value", "N/A") if gdf.brand[i] and gdf.brand[i].get("names") else "N/A",
                    "addresses": gdf.addresses[i] if gdf.addresses[i] else [] # List of address dicts
                }
                places.append(record)
            except Exception as e:
                st.warning(f"Skipping record {gdf.id[i] if hasattr(gdf, 'id') and len(gdf.id) > i else 'unknown ID'} due to parsing error: {e}")
        
        if not places:
            st.warning("No processable POI records were created from the Overture data. This might be due to data format issues or an empty GDF.")
            st.stop()

        pipeline = POIQAPipeline(
            validator=CategoryValidator(),
            completeness=CompletenessChecker(),
            formatter=AddressFormatter(),
            duplicate_detector=DuplicateDetector()
        )
        
        st.info(f"⏳ Running selected QA: {', '.join(selected_qa)}. This may take some time...")
        if pipeline.rate_limit > 0 :
             st.caption(f"ℹ️ Note: The process might pause for 60 seconds if more than {pipeline.rate_limit-1} LLM calls are made, due to API rate limiting.")

        output = pipeline.run(places, selected_qa) 

        st.header("📊 Results")
        results_for_download = {}

        if "qa" in output and output["qa"]:
            st.subheader("QA Verifications")
            qa_df = pd.DataFrame(output["qa"])
            st.dataframe(qa_df)
            results_for_download["qa_verifications"] = qa_df.to_dict(orient='records')
        
        if "duplicates" in output and output["duplicates"]:
            st.subheader("👯 Potential Duplicates")
            if output["duplicates"]:
                for idx, (a, b, reason) in enumerate(output["duplicates"]):
                    st.write(f"   cặp {idx+1}: **{a}** ↔️ **{b}** (Reason: {reason})")
            else:
                st.info("No potential duplicates found based on the selected criteria.")
            results_for_download["potential_duplicates"] = output["duplicates"]
        elif "Duplicate Detection" in selected_qa: # If selected but no duplicates found
            st.subheader("👯 Potential Duplicates")
            st.info("No potential duplicates found.")
            results_for_download["potential_duplicates"] = []


        if results_for_download:
            json_string = json.dumps(results_for_download, indent=4, ensure_ascii=False)
            st.download_button(
                label="📥 Download All Results as JSON",
                data=json_string,
                file_name=f"poi_qa_results_{bbox_str.replace(',', '_').replace(' ', '')}.json",
                mime="application/json"
            )
        else:
            st.info("No results generated to download for the selected QA tasks.")

if __name__ == "__main__":
    streamlit_app()
