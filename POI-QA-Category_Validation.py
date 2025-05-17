# -*- coding: utf-8 -*-
import streamlit as st
import dspy
from pydantic import BaseModel # Required by dspy.Signature
from bs4 import BeautifulSoup
import requests
import time
import json
import traceback

# Overture Maps imports
from overturemaps import core

# --- Configuration & LLM Setup ---
# IMPORTANT: For a deployed app, use st.secrets to store your API key.
# Example: API_KEY = st.secrets["GEMINI_API_KEY"]
# For local development, you can use an environment variable or input field.
# For this example, we'll use a text input in the sidebar for the API key.

st.set_page_config(layout="wide")
st.title("🌍 POI Category Validator")
st.markdown("""
Validate Point of Interest (POI) categories using Overture Maps data, website scraping, 
and a Large Language Model (LLM) via DSPy.
""")

# --- DSPy Signatures and Modules ---
class CategoryValidation(dspy.Signature):
    """Determine if the category of a POI is correct and suggest correction if needed.
    Consider the POI name, its current category, and scraped information from its website.
    If the category seems incorrect, suggest a more appropriate one.
    If correct, confirm it. If unsure, state that.
    Provide a concise response.
    """
    poi_name: str = dspy.InputField(desc="The name of the Point of Interest.")
    poi_description: str = dspy.InputField(desc="Information scraped from the POI's website (e.g., title, meta description, headings).")
    current_category: str = dspy.InputField(desc="The current category assigned to the POI.")
    response: str = dspy.OutputField(desc="Validation result: 'Correct', 'Incorrect, suggest: [new_category]', or 'Unsure'.")

class CategoryValidator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.query = dspy.Predict(CategoryValidation)

    def forward(self, poi_name, poi_description, current_category):
        # Ensure inputs are strings, as dspy can be sensitive
        poi_name_str = str(poi_name) if poi_name is not None else "N/A"
        poi_description_str = str(poi_description) if poi_description is not None else "No description available."
        current_category_str = str(current_category) if current_category is not None else "N/A"
        
        return self.query(
            poi_name=poi_name_str,
            poi_description=poi_description_str,
            current_category=current_category_str
        ).response

# --- Helper Functions ---
@st.cache_data(ttl=3600) # Cache scraping results for 1 hour
def scrape_website(url, timeout=10):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.title.string.strip() if soup.title and soup.title.string else 'No title found'
        
        meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
        meta_desc = meta_desc_tag.get('content', '').strip() if meta_desc_tag else ''
        
        h1_tags = [tag.get_text(strip=True) for tag in soup.find_all('h1')]
        h1_text = '; '.join(h1_tags) if h1_tags else ''

        # Extract some body text, focusing on meaningful paragraphs
        paragraphs = soup.find_all('p')
        body_preview_parts = []
        char_count = 0
        for p in paragraphs:
            p_text = p.get_text(strip=True)
            if p_text:
                body_preview_parts.append(p_text)
                char_count += len(p_text)
                if char_count > 500: # Limit body preview length
                    break
        body_preview = ' '.join(body_preview_parts)

        scraped_text = f"Title: {title}. Meta Description: {meta_desc}. Headings: {h1_text}. Body Preview: {body_preview}"
        return scraped_text.strip()

    except requests.exceptions.Timeout:
        return f"Failed to Scrape (Timeout): {url}"
    except requests.exceptions.RequestException as e:
        return f"Failed to Scrape ({type(e).__name__}): {url}"
    except Exception as e:
        return f"An unexpected error occurred while scraping {url}: {type(e).__name__}"

@st.cache_data(ttl=3600) # Cache Overture Maps data for 1 hour
def get_overture_places(_bbox_tuple):
    try:
        gdf = core.geodataframe("place", bbox=_bbox_tuple)
        return gdf
    except Exception as e:
        st.error(f"Error fetching data from Overture Maps: {e}")
        return None

# --- Streamlit UI ---
# Sidebar for inputs
st.sidebar.header("⚙️ Input Parameters")

# API Key Input
api_key_input = st.sidebar.text_input(
    "🔑 Gemini API Key", 
    type="password", 
    help="Enter your Google AI Studio API Key for Gemini. Your key is not stored."
)

# Bbox input
st.sidebar.subheader("📍 Bounding Box")
min_lon = st.sidebar.number_input("Min Longitude (XMin)", value=-117.0500, format="%.4f")
min_lat = st.sidebar.number_input("Min Latitude (YMin)", value=33.0500, format="%.4f")
max_lon = st.sidebar.number_input("Max Longitude (XMax)", value=-117.0000, format="%.4f")
max_lat = st.sidebar.number_input("Max Latitude (YMax)", value=33.1000, format="%.4f")

# Max POIs to process
max_pois_to_process_all = st.sidebar.slider(
    "Max POIs to Process", 0, 500, 10, 
    help="Set to 0 to process all found POIs (can be slow and costly)."
)

# Rate Limiting Configuration
st.sidebar.subheader("⏱️ API Rate Limiting")
calls_before_pause = st.sidebar.slider("LLM Calls Before Pause", 1, 30, 10, help="Number of LLM calls before a pause.")
pause_duration = st.sidebar.slider("Pause Duration (seconds)", 10, 120, 60, help="Duration of pause to respect API limits.")


if st.sidebar.button("🚀 Validate POIs", use_container_width=True):
    if not api_key_input:
        st.error("❌ Please enter your Gemini API Key in the sidebar.")
        st.stop()

    try:
        lm = dspy.LM('gemini/gemini-2.0-flash', api_key=api_key_input)
        dspy.configure(lm=lm)
        st.success("✅ LLM Configured with Gemini API Key.")
    except Exception as e:
        st.error(f"❌ Failed to initialize LLM. Check your API key. Error: {e}")
        st.stop()

    bbox_input = (min_lon, min_lat, max_lon, max_lat)
    if not (bbox_input[0] < bbox_input[2] and bbox_input[1] < bbox_input[3]):
        st.error("❌ Invalid bounding box: Min coordinates must be less than Max coordinates.")
        st.stop()

    all_results_data = []
    
    main_status = st.status(f"Processing POIs for Bbox: {bbox_input}...", expanded=True)
    
    progress_bar_container = main_status.empty()
    current_poi_status_text = main_status.empty()
    results_log_container = main_status.container(height=300) # Scrollable log

    try:
        main_status.write("Fetching data from Overture Maps...")
        gdf = get_overture_places(bbox_input)

        if gdf is None or gdf.empty:
            main_status.warning("No places found for the given bounding box or error fetching data.")
            st.stop()

        main_status.write(f"Found {len(gdf)} places in the bounding box.")

        validator = CategoryValidator()

        num_to_process = len(gdf)
        if max_pois_to_process_all > 0:
            num_to_process = min(len(gdf), max_pois_to_process_all)
            main_status.write(f"Processing up to {num_to_process} POIs based on user setting.")
        
        if num_to_process == 0:
            main_status.info("No POIs to process.")
            st.stop()

        api_call_counter = 0

        for i_places in range(num_to_process):
            # API Rate Limiting
            if api_call_counter > 0 and api_call_counter % calls_before_pause == 0:
                current_poi_status_text.info(f"Pausing for {pause_duration}s (API rate limit)... {api_call_counter} calls made.")
                time.sleep(pause_duration)

            place_data = gdf.iloc[i_places]
            
            current_poi_name = "N/A"
            if place_data.get('names') and isinstance(place_data['names'], dict) and 'primary' in place_data['names']:
                current_poi_name = place_data['names']['primary']
            elif place_data.get('names') and isinstance(place_data['names'], dict) and 'common' in place_data['names'] and place_data['names']['common']:
                 current_poi_name = place_data['names']['common'][0]
            else:
                current_poi_name = f"Unnamed POI (ID: {place_data.get('id', 'Unknown')})"

            current_poi_status_text.text(f"POI {i_places + 1}/{num_to_process}: {current_poi_name}")
            progress_bar_container.progress((i_places + 1) / num_to_process)

            # Scrape website
            scraped_description = "No website information available or not applicable."
            website_url = None
            websites_list = place_data.get('websites')
            if websites_list and isinstance(websites_list, list) and len(websites_list) > 0:
                website_url = websites_list[0] 
                if website_url and isinstance(website_url, str) and website_url.startswith("http"):
                    results_log_container.write(f" scraping {website_url}...")
                    scraped_description = scrape_website(website_url)
                    results_log_container.write(f"  -> Scraped: {scraped_description[:100]}...")
                else:
                    scraped_description = f"Invalid or missing website URL: {website_url}"
                    results_log_container.write(f"  -> {scraped_description}")
            
            current_cat = "N/A"
            categories_data = place_data.get('categories')
            if categories_data and isinstance(categories_data, dict) and 'primary' in categories_data:
                current_cat = categories_data['primary']
            elif categories_data and isinstance(categories_data, dict) and 'alternate' in categories_data and categories_data['alternate']:
                current_cat = categories_data['alternate'][0]

            # LLM Validation
            llm_response_text = "LLM call skipped or failed."
            try:
                results_log_container.write(f" validating with LLM...")
                llm_response = validator(
                    poi_name=current_poi_name,
                    poi_description=scraped_description,
                    current_category=current_cat
                )
                llm_response_text = llm_response
                api_call_counter += 1
                results_log_container.write(f"  -> LLM says: {llm_response_text}")
            except Exception as e:
                llm_response_text = f"LLM Error: {str(e)[:200]}"
                results_log_container.error(f"LLM validation error for {current_poi_name}: {e}")

            result_entry = {
                "overture_id": place_data.get('id', 'Unknown'),
                "poi_name": current_poi_name,
                "website_url": website_url if website_url else "N/A",
                "scraped_content_summary": scraped_description,
                "current_category": current_cat,
                "llm_validation_response": llm_response_text
            }
            all_results_data.append(result_entry)

        main_status.update(label="🎉 Processing complete!", state="complete", expanded=False)
        
        st.subheader("📊 Results Summary")
        st.dataframe(all_results_data)

        if all_results_data:
            json_string = json.dumps(all_results_data, indent=4)
            # Sanitize bbox for filename
            bbox_str = "_".join(map(str, bbox_input)).replace(".", "p")
            st.download_button(
                label="📥 Download All Results as JSON",
                data=json_string,
                file_name=f"poi_validation_results_bbox_{bbox_str}.json",
                mime="application/json",
                use_container_width=True
            )

    except requests.exceptions.Timeout:
        main_status.error("A website scraping request timed out. Please try again or adjust timeout in code.")
    except Exception as e:
        main_status.error(f"An unexpected error occurred: {e}")
        st.error(traceback.format_exc())

else:
    st.info("Adjust parameters in the sidebar and click '🚀 Validate POIs' to begin.")

st.markdown("---")
st.markdown("Built with ❤️ by Chaavan, Cyrus and Copilot using Streamlit, Overture Maps, and DSPy.")