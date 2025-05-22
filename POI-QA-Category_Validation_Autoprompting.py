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
from dspy.teleprompt import MIPROv2 # Added for MIPROv2

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DEFAULT_FREE_GEMINI_API_KEY = "AIzaSyC_YvqPom_7xgJCt2V6QLSH8LQpsbH9dzM" # Replace if necessary

# ─── DSPY & LLM SETUP ─────────────────────────────────────────────────────────
lm = None # Will be configured in Streamlit app

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

def scrape_place_website(_place_name: str, urls: List[str]) -> str:
    """Scrape the first URL in the `urls` list, if any."""
    if not urls:
        return ""
    first_url = urls[0]
    if not first_url or not isinstance(first_url, str):
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
class CompletenessChecker:
    def forward(self, record: Dict[str, Any], required_fields: List[str]) -> List[str]:
        missing: List[str] = []
        for field in required_fields:
            if not bool (record.get(field)):
                missing.append(field)
        return missing



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
    CALLS_BEFORE_PAUSE_FREE = 15
    PAUSE_DURATION_SECONDS = 60

    def __init__(
            self,
            validator: CategoryValidator, 
            completeness: CompletenessChecker, 
            formatter: AddressFormatter, 
            duplicate_detector: DuplicateDetector, 
            api_mode: str
            ):
        self.validator = validator
        self.completeness = completeness
        self.formatter = formatter
        self.duplicate_detector = duplicate_detector
        self.api_mode = api_mode
        self.call_count = 0
        if self.api_mode == "Free (Rate Limited)":
            self.rate_limit = POIQAPipeline.CALLS_BEFORE_PAUSE_FREE
        else:
            self.rate_limit = 0 # No script-enforced limit for paid mode

    def _api_call(self, fn, *args):
        if not dspy.settings.lm:
            st.error("LLM not configured. Cannot make API call.")
            raise Exception("LLM not configured for API call.")
        result = fn(*args)
        self.call_count += 1
        if self.rate_limit > 0 and self.call_count > 0 and self.call_count % self.rate_limit == 0:
            # Use a placeholder in the main area for the pause message if sidebar is not suitable
            main_area_placeholder = st.empty()
            main_area_placeholder.warning(
                f"⚠️ API call limit ({self.rate_limit} calls for Free mode) reached. "
                f"Pausing for {POIQAPipeline.PAUSE_DURATION_SECONDS}s …"
            )
            time.sleep(POIQAPipeline.PAUSE_DURATION_SECONDS)
            main_area_placeholder.empty()
        return result

    def validate_one(self, record: Dict[str, Any], selected_qa: List[str]) -> Dict[str, Any]:
        name       = record.get("name", "")
        category   = record.get("category", "")
        urls       = record.get("websites", [])
        addresses  = record.get("addresses", [])
        result = {
            "name": name,
            "category_suggestion": "Not performed",
            "missing_fields": "Not performed",
            "corrected_address": "Not performed"
        }
        if "Category Validation" in selected_qa:
            desc = scrape_place_website(name, urls)
            cat_suggestion = self._api_call(self.validator.forward, name, desc, category)
            result["category_suggestion"] = cat_suggestion
        if "Completeness Check" in selected_qa:
            required = ["name","category","websites","socials","emails","phones","addresses"]
            print(type(self.completeness))
            missing = self.completeness.forward(record, required)
            result["missing_fields"] = missing if missing else []
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
                res = self._api_call(self.duplicate_detector.forward, rec1_simple, rec2_simple)
                if res.get("duplicate"):
                    dups.append((places[i]["name"], places[i+1+j]["name"], res.get("reason", "No reason provided")))
        return dups

    def run(self, places: List[dict], selected_qa: List[str]) -> Dict[str, Any]:
        qa_results = []
        single_record_qa_selected = any(item in selected_qa for item in ["Category Validation", "Completeness Check", "Address Formatting"])
        
        progress_bar_placeholder = st.empty()
        status_text_placeholder = st.empty()

        if single_record_qa_selected:
            progress_bar = progress_bar_placeholder.progress(0)
            for i, rec in enumerate(places):
                status_text_placeholder.text(f"Validating POI {i+1}/{len(places)}: {rec.get('name', 'Unnamed POI')}")
                qa_results.append(self.validate_one(rec, selected_qa))
                progress_bar.progress((i + 1) / len(places))
            status_text_placeholder.text(f"Validated {len(places)} POIs.")
            # Clear progress bar and status text after completion
            time.sleep(2) # Keep message for a bit
            progress_bar_placeholder.empty()
            status_text_placeholder.empty()

        duplicates_result = []
        if "Duplicate Detection" in selected_qa:
            with st.spinner("Detecting duplicates... This may take a while for many POIs."):
                duplicates_result = self.detect_duplicates(places)
        
        return {"qa": qa_results, "duplicates": duplicates_result}

# ─── STREAMLIT APP ───────────────────────────────────────────────────────────
def streamlit_app():
    st.set_page_config(layout="wide")
    st.title("🌍 Overture POI Quality Assurance Tool")

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("⚙️ Configuration")
    api_mode = st.sidebar.radio(
        "Select API Key Mode:",
        ("Free (Rate Limited)", "Paid (Your API Key)"),
        help="Free mode uses a pre-configured key (if set in code) and is rate-limited. Paid mode uses your own key."
    )
    api_key_to_use = None
    user_provided_api_key = ""

    if api_mode == "Free (Rate Limited)":
        api_key_to_use = DEFAULT_FREE_GEMINI_API_KEY
        st.sidebar.info(
            f"Using default API key. Process will pause for {POIQAPipeline.PAUSE_DURATION_SECONDS}s "
            f"after every {POIQAPipeline.CALLS_BEFORE_PAUSE_FREE} LLM calls."
        )
        if DEFAULT_FREE_GEMINI_API_KEY == "YOUR_DEFAULT_FREE_API_KEY_PLACEHOLDER": 
            st.sidebar.warning("Default 'Free' API key is a placeholder. Replace it in the script.")
    elif api_mode == "Paid (Your API Key)":
        user_provided_api_key = st.sidebar.text_input("Gemini API Key", type="password", value="")
        if user_provided_api_key:
            api_key_to_use = user_provided_api_key
            st.sidebar.info("Using your API key. No script-enforced rate limit.")

    global lm
    llm_configured_successfully = False
    if 'dspy_configured_with_key' not in st.session_state:
        st.session_state.dspy_configured_with_key = None
    if 'cached_lms' not in st.session_state:
        st.session_state.cached_lms = {}

    if api_key_to_use:
        if api_mode == "Free (Rate Limited)" and api_key_to_use == "YOUR_DEFAULT_FREE_API_KEY_PLACEHOLDER":
            llm_configured_successfully = False 
        else:
            current_lm_instance = None
            try:
                if api_key_to_use in st.session_state.cached_lms:
                    current_lm_instance = st.session_state.cached_lms[api_key_to_use]
                else:
                    # Ensure max_tokens is appropriate for Gemini Flash 2.0 (or the model used)
                    current_lm_instance = dspy.LM('gemini/gemini-2.0-flash', api_key=api_key_to_use, max_tokens=2048) 
                    st.session_state.cached_lms[api_key_to_use] = current_lm_instance
                
                if dspy.settings.lm is not current_lm_instance:
                    dspy.configure(lm=current_lm_instance)
                
                lm = current_lm_instance # Set the global lm
                st.session_state.dspy_configured_with_key = api_key_to_use
                st.sidebar.success(f"LLM Configured ({api_mode})!")
                llm_configured_successfully = True
            except Exception as e:
                key_snippet = api_key_to_use[:5] + "..." if api_key_to_use and len(api_key_to_use) > 5 else api_key_to_use
                st.sidebar.error(f"LLM Config Error ({api_mode}, key: {key_snippet}): {e}")
                if api_key_to_use in st.session_state.cached_lms: del st.session_state.cached_lms[api_key_to_use]
                if st.session_state.dspy_configured_with_key == api_key_to_use: st.session_state.dspy_configured_with_key = None
                if dspy.settings.lm is current_lm_instance and current_lm_instance is not None: dspy.settings.lm = None
                lm = None
                llm_configured_successfully = False
    else:
        if api_mode == "Paid (Your API Key)" and not user_provided_api_key:
            st.sidebar.warning("Please enter your Gemini API Key for 'Paid' mode.")
        llm_configured_successfully = False

    # --- MIPROv2 Compilation Logic (in sidebar) ---
    st.sidebar.subheader("Module Compilation")
    if 'compiled_modules' not in st.session_state:
        st.session_state.compiled_modules = {}
    if 'last_compiled_with_key' not in st.session_state:
        st.session_state.last_compiled_with_key = None

    compiled_category_validator = None
    # compiled_completeness_checker is not a DSPy module, will be instantiated directly
    compiled_address_formatter = None
    compiled_duplicate_detector = None

    if llm_configured_successfully and current_lm_instance:
        # Define mipro_prompt_model and mipro_task_model using the configured lm
        mipro_prompt_model = current_lm_instance
        mipro_task_model = current_lm_instance

        if st.session_state.last_compiled_with_key == api_key_to_use and \
           api_key_to_use in st.session_state.compiled_modules and \
           all(module is not None for module in st.session_state.compiled_modules[api_key_to_use].values()):
            st.sidebar.info("Using cached compiled QA modules.")
            modules = st.session_state.compiled_modules[api_key_to_use]
            compiled_category_validator = modules.get('validator')
            # compiled_completeness_checker is instantiated directly
            compiled_address_formatter = modules.get('formatter')
            compiled_duplicate_detector = modules.get('duplicate_detector')
        else:
            with st.sidebar.status("Compiling QA modules with MIPROv2...", expanded=False) as compile_status:
                st.write("Defining metrics and trainsets...")
                metric_cat_val = lambda gold, pred, trace: getattr(gold, 'response', None) == getattr(pred, 'response', None)
                metric_comp_check = lambda gold, pred, trace: getattr(gold, 'missing_fields', None) == getattr(pred, 'missing_fields', None)
                metric_addr_fmt = lambda gold, pred, trace: getattr(gold, 'corrected', None) == getattr(pred, 'corrected', None)
                metric_dup_detect = lambda gold, pred, trace: getattr(gold, 'is_duplicate', None) == getattr(pred, 'is_duplicate', None)
                
                train_cat_val = [
                    dspy.Example(poi_name="Dummy Cafe", poi_description="A place to get coffee and pastries.", current_category="food/cafe", response="food/cafe").with_inputs("poi_name", "poi_description", "current_category"),
                    dspy.Example(poi_name="Tech Solutions Inc.", poi_description="Software development and IT consulting.", current_category="services/consulting", response="office/it_services").with_inputs("poi_name", "poi_description", "current_category"),
                    dspy.Example(poi_name="Greenwood Park", poi_description="Public park with playground and trails.", current_category="amenity/park", response="leisure/park").with_inputs("poi_name", "poi_description", "current_category"),
                    dspy.Example(poi_name="The Book Nook", poi_description="Independent bookstore selling new and used books.", current_category="retail/books", response="shop/books").with_inputs("poi_name", "poi_description", "current_category"),
                    dspy.Example(poi_name="City General Hospital", poi_description="Full-service hospital with emergency room.", current_category="healthcare", response="amenity/hospital").with_inputs("poi_name", "poi_description", "current_category"),
                    dspy.Example(poi_name="Luigi's Pizzeria", poi_description="Authentic Italian pizza and pasta.", current_category="restaurant", response="food/restaurant").with_inputs("poi_name", "poi_description", "current_category"),
                    dspy.Example(poi_name="Quick Stop", poi_description="Convenience store open 24/7.", current_category="shop", response="shop/convenience").with_inputs("poi_name", "poi_description", "current_category"),
                    dspy.Example(poi_name="Grand Cinema", poi_description="Multiplex cinema showing latest movies.", current_category="entertainment", response="amenity/cinema").with_inputs("poi_name", "poi_description", "current_category"),
                    dspy.Example(poi_name="Central Library", poi_description="Public library with large collection of books and digital resources.", current_category="public_service/library", response="amenity/library").with_inputs("poi_name", "poi_description", "current_category"),
                    dspy.Example(poi_name="AutoCare Garage", poi_description="Car repair and maintenance services.", current_category="automotive", response="shop/car_repair").with_inputs("poi_name", "poi_description", "current_category")
                ]
                train_comp_check = [
                    dspy.Example(record={"name": "Store", "category": "shop/books", "websites": ["http://example.com"]}, required_fields=["name", "category", "websites", "addresses"], missing_fields=["addresses"]).with_inputs("record", "required_fields"),
                    dspy.Example(record={"name": "Restaurant X", "category": "food/restaurant", "addresses": [{"freeform":"123 Main St"}]}, required_fields=["name", "category", "websites", "phones"], missing_fields=["websites", "phones"]).with_inputs("record", "required_fields"),
                    dspy.Example(record={"name": "Park Y", "category": "leisure/park", "websites": ["http://parky.com"], "addresses": [{"freeform":"456 Oak Ave"}], "phones":["555-1234"]}, required_fields=["name", "category", "websites", "addresses", "phones"], missing_fields=[]).with_inputs("record", "required_fields"),
                    dspy.Example(record={"category": "shop/electronics", "websites": ["http://electronics.com"]}, required_fields=["name", "category", "websites"], missing_fields=["name"]).with_inputs("record", "required_fields"),
                    dspy.Example(record={"name": "Coffee Place"}, required_fields=["name", "category", "addresses"], missing_fields=["category", "addresses"]).with_inputs("record", "required_fields"),
                    dspy.Example(record={}, required_fields=["name", "category"], missing_fields=["name", "category"]).with_inputs("record", "required_fields"),
                    dspy.Example(record={"name": "Gym Z", "category": "leisure/fitness_centre", "socials": ["http://fb.com/gymz"]}, required_fields=["name", "category", "addresses", "phones"], missing_fields=["addresses", "phones"]).with_inputs("record", "required_fields"),
                    dspy.Example(record={"name": "Hotel Alpha", "category": "lodging/hotel", "emails": ["contact@hotelalpha.com"]}, required_fields=["name", "category", "addresses", "websites"], missing_fields=["addresses", "websites"]).with_inputs("record", "required_fields"),
                    dspy.Example(record={"name": "Museum Beta", "addresses": [{"freeform":"789 Museum Rd"}], "phones": ["555-0011"]}, required_fields=["name", "category", "websites"], missing_fields=["category", "websites"]).with_inputs("record", "required_fields"),
                    dspy.Example(record={"name": "Bakery Charlie", "category": "food/bakery", "websites": ["http://bakery.com"], "addresses": [{"freeform":"101 Sweet Ln"}], "phones":["555-빵빵"], "emails":["info@bakery.com"], "socials":[]}, required_fields=["name", "category", "websites", "addresses", "phones", "emails", "socials"], missing_fields=[]).with_inputs("record", "required_fields")
                ]
                train_addr_fmt = [
                    dspy.Example(address="123 main st anytown", corrected="123 Main St, Anytown").with_inputs("address"),
                    dspy.Example(address="456 OAK AVENUE, SUITE 100, big city, ca 90210", corrected="456 Oak Avenue, Suite 100, Big City, CA 90210").with_inputs("address"),
                    dspy.Example(address="789 pine ln. smallville", corrected="789 Pine Ln., Smallville").with_inputs("address"),
                    dspy.Example(address="PO Box 1234, Somewhere, ST 54321", corrected="PO Box 1234, Somewhere, ST 54321").with_inputs("address"), # Already good
                    dspy.Example(address="elm street 1 apt 2b", corrected="1 Elm Street, Apt 2B").with_inputs("address"), # Number first
                    dspy.Example(address=" rural route 5, box 88, country town ", corrected="Rural Route 5, Box 88, Country Town").with_inputs("address"),
                    dspy.Example(address="221b baker street london", corrected="221B Baker Street, London").with_inputs("address"),
                    dspy.Example(address="One Infinite Loop, Cupertino, CA 95014", corrected="One Infinite Loop, Cupertino, CA 95014").with_inputs("address"), # Already good
                    dspy.Example(address="1600 pennsylvania ave nw washington dc 20500", corrected="1600 Pennsylvania Ave NW, Washington, DC 20500").with_inputs("address"),
                    dspy.Example(address="Hauptstrasse 10, Berlin, 10115", corrected="Hauptstrasse 10, Berlin, 10115").with_inputs("address") # International example
                ]
                train_dup_detect = [
                    dspy.Example(rec1={"name": "Place A", "address": "1 Street", "category": "food"}, rec2={"name": "Place A", "address": "1 Street", "category": "food"}, is_duplicate=True, reason="Identical name, address, and category.").with_inputs("rec1", "rec2"),
                    dspy.Example(rec1={"name": "Cafe Alpha", "address": "123 Main St"}, rec2={"name": "Cafe Beta", "address": "456 Oak St"}, is_duplicate=False, reason="Different names and addresses.").with_inputs("rec1", "rec2"),
                    dspy.Example(rec1={"name": "The Coffee Shop", "address": "10 Market Square", "website": "coffee.com"}, rec2={"name": "Coffee Shop, The", "address": "10 Market Sq.", "website": "coffee.com"}, is_duplicate=True, reason="Similar name, address, and same website.").with_inputs("rec1", "rec2"),
                    dspy.Example(rec1={"name": "Pizza Place", "address": "Unit 5, Retail Park"}, rec2={"name": "Pizza Place", "address": "Unit 7, Retail Park"}, is_duplicate=False, reason="Same name but different unit number in address.").with_inputs("rec1", "rec2"),
                    dspy.Example(rec1={"name": "Bookstore", "address": "Online only", "website": "books.example"}, rec2={"name": "My Bookstore", "address": "Online only", "website": "books.example"}, is_duplicate=True, reason="Similar name, online only, and same website.").with_inputs("rec1", "rec2"),
                    dspy.Example(rec1={"name": "Dr. Smith Clinic", "address": "1 Health Rd", "phone": "555-0100"}, rec2={"name": "Dr. Smith's Office", "address": "1 Health Road", "phone": "555-0100"}, is_duplicate=True, reason="Name variation, address variation, but same core details and phone.").with_inputs("rec1", "rec2"),
                    dspy.Example(rec1={"name": "Global Corp HQ", "address": "1 Corporate Dr"}, rec2={"name": "Global Corp Branch", "address": "2 Business Ave"}, is_duplicate=False, reason="Similar name prefix but clearly different locations/types (HQ vs Branch).").with_inputs("rec1", "rec2"),
                    dspy.Example(rec1={"name": "Park View Hotel", "address": "10 Park Rd"}, rec2={"name": "Park View Apartments", "address": "10 Park Rd"}, is_duplicate=False, reason="Same address but different type of establishment (Hotel vs Apartments).").with_inputs("rec1", "rec2"),
                    dspy.Example(rec1={"name": "Starbucks", "address": "Main St & 1st Ave"}, rec2={"name": "Starbucks Coffee", "address": "Corner of Main and First"}, is_duplicate=True, reason="Common name, address describes same intersection.").with_inputs("rec1", "rec2"),
                    dspy.Example(rec1={"name": "Closed Restaurant", "address": "Old Address St", "status": "permanently_closed"}, rec2={"name": "New Restaurant", "address": "Old Address St", "status": "operating"}, is_duplicate=False, reason="Same address but one is closed and the other is a new, operating entity.").with_inputs("rec1", "rec2")
                ]

                try:
                    st.write("Compiling Category Validator...")
                    category_validator_module = CategoryValidator()
                    teleprompter_cat = MIPROv2(prompt_model=mipro_prompt_model, task_model=mipro_task_model, metric=metric_cat_val, auto='light') # Adjusted MIPRO params
                    compiled_category_validator = teleprompter_cat.compile(category_validator_module, trainset=train_cat_val, requires_permission_to_run=False)
                    st.write("Category Validator compiled.")

                    # CompletenessChecker is a plain Python class, no compilation needed.
                    # It will be instantiated directly.

                    st.write("Compiling Address Formatter...")
                    address_formatter_module = AddressFormatter()
                    teleprompter_addr = MIPROv2(prompt_model=mipro_prompt_model, task_model=mipro_task_model, metric=metric_addr_fmt, auto='light')
                    compiled_address_formatter = teleprompter_addr.compile(address_formatter_module, trainset=train_addr_fmt, requires_permission_to_run=False)
                    st.write("Address Formatter compiled.")

                    st.write("Compiling Duplicate Detector...")
                    duplicate_detector_module = DuplicateDetector()
                    teleprompter_dup = MIPROv2(prompt_model=mipro_prompt_model, task_model=mipro_task_model, metric=metric_dup_detect, auto='light')
                    compiled_duplicate_detector = teleprompter_dup.compile(duplicate_detector_module, trainset=train_dup_detect, requires_permission_to_run=False)
                    st.write("Duplicate Detector compiled.")

                    st.session_state.compiled_modules[api_key_to_use] = {
                        'validator': compiled_category_validator,
                        'formatter': compiled_address_formatter,}
                    st.session_state.compiled_modules[api_key_to_use] = {
                        'validator': compiled_category_validator,
                        # 'completeness' is not stored as it's not compiled
                        'formatter': compiled_address_formatter,
                        'duplicate_detector': compiled_duplicate_detector
                    }
                    st.session_state.last_compiled_with_key = api_key_to_use

                except Exception as e:
                    st.sidebar.error(f"Error compiling modules: {e}")
                    if api_key_to_use in st.session_state.compiled_modules:
                        del st.session_state.compiled_modules[api_key_to_use]
                    if st.session_state.last_compiled_with_key == api_key_to_use:
                        st.session_state.last_compiled_with_key = None
                    if st.session_state.last_compiled_with_key == api_key_to_use:
                        st.session_state.last_compiled_with_key = None
                    compiled_category_validator, compiled_address_formatter, compiled_duplicate_detector = None, None, None
                    # compiled_completeness_checker is not part of this group
                    compile_status.update(label=f"Compilation failed: {e}", state="error", expanded=True)
    
    if not llm_configured_successfully :
        st.sidebar.error("LLM is not configured. Please check API key settings. Operations requiring LLM will not be available.")
        # Check only for modules that are compiled
        if not (compiled_category_validator and compiled_address_formatter and compiled_duplicate_detector):
             # No st.stop() here, let the user see the main page but the button will be disabled or show errors.
             pass
    # --- SIDEBAR INPUTS ---
    st.sidebar.header("📍 Bounding Box Input")
    default_bbox = "-117.05, 33.00, -117.00, 33.05" # Example: San Diego area
    bbox_str = st.sidebar.text_input("Enter Bounding Box (min_lon, min_lat, max_lon, max_lat):", default_bbox)

    st.sidebar.header("🔍 QA Verifications")
    available_qa = ["Category Validation", "Completeness Check", "Address Formatting", "Duplicate Detection"]
    selected_qa = st.sidebar.multiselect("Select QA Verifications to perform:", available_qa, default=available_qa)

    # CompletenessChecker is not compiled, so it's not checked here for compilation status
    run_button_disabled = not (llm_configured_successfully and 
                               (not ("Category Validation" in selected_qa) or compiled_category_validator) and
                               (not ("Address Formatting" in selected_qa) or compiled_address_formatter) and
                               (not ("Duplicate Detection" in selected_qa) or compiled_duplicate_detector))


    if st.sidebar.button("🚀 Run QA on POIs", type="primary", disabled=run_button_disabled):
        if not bbox_str: st.error("❗ Please enter a bounding box in the sidebar."); st.stop()
        if not selected_qa: st.error("❗ Please select at least one QA verification in the sidebar."); st.stop()
        if not dspy.settings.lm: st.error("❗ LLM not configured. Check API Key in the sidebar."); st.stop()

        modules_ready = True
        if "Category Validation" in selected_qa and not compiled_category_validator: modules_ready = False
        if "Address Formatting" in selected_qa and not compiled_address_formatter: modules_ready = False
        if "Duplicate Detection" in selected_qa and not compiled_duplicate_detector: modules_ready = False
        
        if not modules_ready:
            st.error("❗ Not all selected QA modules are compiled. Check LLM configuration and compilation status in sidebar.")
            st.stop()

        try:
            bbox_parts = [float(p.strip()) for p in bbox_str.split(',')]
            if len(bbox_parts) != 4: raise ValueError("BBOX must have 4 parts.")
            bbox = tuple(bbox_parts)
        except ValueError as e:
            st.error(f"Invalid BBOX format: {e}")
            st.stop()

        st.info(f"Fetching POIs for BBOX: {bbox_str} (Note: Using dummy data for now)")
        # Actual data fetching (replace with Overture SDK call)
        try:
            # entity_df = core.Place.within_bbox(bbox,ประเภท='place') # Example, adjust as needed
            # places = entity_df.to_dict(orient='records')
            # if not places:
            #     st.warning("No POIs found for the given bounding box. Using dummy data.")
            #     raise ValueError("No POIs from SDK")
            # st.success(f"Fetched {len(places)} POIs from Overture Maps.")
            # For now, always use dummy data as SDK integration might be complex/slow for quick tests
            raise ValueError("Using dummy data as placeholder")

        except Exception as e: # Catch SDK errors or the ValueError above
            st.warning(f"Warning: POI data fetching from Overture SDK failed or not implemented ({e}). Using dummy data for demonstration.")
            places = [
                dspy.Example(id="1", name="Dummy POI Alpha", category="shop/books", websites=["http://example.com/alpha"], addresses=[{"freeform": "123 Global St, World"}]).toDict(),
                dspy.Example(id="2", name="Dummy POI Beta", category="food/restaurant", websites=["http://example.com/beta"], addresses=[{"freeform": "456 Universal Ave, Cosmos"}]).toDict(),
                dspy.Example(id="3", name="Dummy POI Alpha", category="shop/books", websites=["http://example.com/alpha_dup"], addresses=[{"freeform": "123 Global St, World"}]).toDict(),
                dspy.Example(id="4", name="Tech Store Gamma", category="shop/electronics", websites=[], addresses=[{"freeform": "789 Circuit Board, Silicon Valley"}], phones=["555-0100"], emails=["contact@techgamma.com"]).toDict(),
                dspy.Example(id="5", name="Park Omega", category="park", websites=["http://omegapark.org"], addresses=[]).toDict(), # Missing address
            ]
        pipeline = POIQAPipeline(
            validator=compiled_category_validator,
            completeness=CompletenessChecker(), # Instantiate directly
            formatter=compiled_address_formatter,
            duplicate_detector=compiled_duplicate_detector,
            api_mode=api_mode
        )

        output = pipeline.run(places, selected_qa,
            duplicate_detector=compiled_duplicate_detector,
            api_mode=api_mode
        )

        output = pipeline.run(places, selected_qa)

        # --- MAIN AREA FOR OUTPUTS ---
        st.header("📊 Results")
        results_for_download = {} # Initialize here

        if "qa" in output and output["qa"]:
            st.subheader("QA Verifications")
            qa_df = pd.DataFrame(output["qa"])
            st.dataframe(qa_df)
            results_for_download["qa_verifications"] = qa_df.to_dict(orient='records')
        
        if "duplicates" in output and output["duplicates"]:
            st.subheader("👯 Potential Duplicates")
            if output["duplicates"]:
                for idx, (a, b, reason) in enumerate(output["duplicates"]):
                    st.write(f"   Pair {idx+1}: **{a}** ↔️ **{b}** (Reason: {reason})")
            else: 
                st.info("No potential duplicates found.")
            results_for_download["potential_duplicates"] = output["duplicates"]
        elif "Duplicate Detection" in selected_qa: # Ensure this section appears even if no duplicates are found but task was selected
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
            st.info("No results generated or selected QA tasks did not produce downloadable output.")
    elif run_button_disabled and not (llm_configured_successfully and lm):
         st.warning("LLM is not configured or modules are not compiled. Please check settings in the sidebar. QA run is disabled.")
    elif run_button_disabled:
         st.warning("QA modules are not compiled. Please check compilation status in the sidebar. QA run is disabled.")


if __name__ == "__main__":
    streamlit_app()
