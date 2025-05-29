# src/acquire.py

import requests
from bs4 import BeautifulSoup
import time
import os
from urllib.parse import urljoin # No longer need quote for topic
from datetime import datetime

# --- Configuration ---
BASE_URL = "https://www.uscis.gov"
DECISIONS_PAGE_PATH = "/administrative-appeals/aao-decisions/aao-non-precedent-decisions" # This is the base page for queries

# Target parameters using the discovered query parameter values
TARGET_URI_1 = "19"  # For "I-140 - Immigrant Petition for Alien Worker (Extraordinary Ability)"
TARGET_M = "2"       # For "February"
TARGET_Y = "1"       # For "2025" (assuming '1' consistently maps to the latest year, which is 2025 in the dropdown)

# For display/logging purposes, let's keep the text versions
TARGET_TOPIC_TEXT_DISPLAY = "I-140 - Immigrant Petition for Alien Worker (Extraordinary Ability)"
TARGET_MONTH_DISPLAY = "February"
TARGET_YEAR_DISPLAY = "2025 (mapped by y=1)"


# Output directory for raw PDFs
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Polite crawling
CRAWL_DELAY_SECONDS = 10 # From robots.txt
USER_AGENT = "LegalRAGInternshipProject/1.0 (Contact: your_email@example.com; for academic research)" # Replace with your email

# --- Helper Functions ---
def construct_filtered_url(uri_1_val, m_val, y_val, page=0, items_per_page=10):
    """Constructs the URL for a specific filtered page using query parameters."""
    base_page_url = urljoin(BASE_URL, DECISIONS_PAGE_PATH)
    
    params = {
        'uri_1': uri_1_val,
        'm': m_val,
        'y': y_val,
        'items_per_page': items_per_page,
        'page': page  # Page is 0-indexed for the query parameter
    }
    
    # Create query string
    query_string = "&".join(f"{key}={value}" for key, value in params.items())
    return f"{base_page_url}?{query_string}"

def make_request(url):
    """Makes a request with appropriate headers and delay."""
    print(f"Requesting URL: {url}")
    headers = {'User-Agent': USER_AGENT}
    try:
        time.sleep(CRAWL_DELAY_SECONDS) # Adhere to Crawl-delay
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status() # Raise an exception for HTTP errors
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error during request to {url}: {e}")
        return None

def parse_decision_links(html_content):
    """Parses HTML to find PDF links and their publication dates."""
    # Try with both parsers if one fails, starting with lxml
    try:
        soup = BeautifulSoup(html_content, 'lxml')
    except Exception as e_lxml:
        print(f"lxml parser failed: {e_lxml}. Trying html.parser.")
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
        except Exception as e_html:
            print(f"html.parser also failed: {e_html}. Cannot parse HTML.")
            return []

    decision_links = []

    # Attempt 1: Find the specific 'view-content' div
    # Drupal class names can sometimes have extra spaces or be part of a multi-class string.
    # We can also try finding by ID if one is known and stable.
    # For now, let's stick to class and make the search a bit more robust.
    
    # Try finding 'view-content'
    # Common ways it might appear: 'view-content', ' views-content ', 'region-content view-content' etc.
    # BeautifulSoup's class_ selector handles multi-class attributes well if 'view-content' is one of them.
    results_container = soup.find('div', class_='view-content')

    if not results_container:
        print("Warning: Could not find 'div.view-content'. Searching for 'div.views-row' in the whole document as a fallback.")
        # Fallback: search for 'views-row' divs in the entire document
        item_divs = soup.find_all('div', class_='views-row')
        if not item_divs:
            print("Error: No 'div.view-content' found, and no 'div.views-row' found in the entire document.")
            return []
    else:
        print("Successfully found 'div.view-content'. Searching for 'div.views-row' within it.")
        item_divs = results_container.find_all('div', class_='views-row')
        if not item_divs:
            print("Error: Found 'div.view-content', but no 'div.views-row' found within it.")
            return []

    print(f"Found {len(item_divs)} potential decision entries (div.views-row).")

    for item_div in item_divs: # item_div is a single 'div.views-row'
        link_tag = None
        pdf_url = None
        publication_date_str = "Unknown"
        link_text_str = "Unknown"

        # 1. Extract Link - Search for the <a> tag directly within item_div
        #    This is a more direct approach if the intermediate 'views-field-field-content' div
        #    is not present or has a different class.
        link_tag = item_div.find('a', href=lambda href: href and href.endswith('.pdf'))
        
        if link_tag and link_tag.has_attr('href'):
            relative_pdf_url = link_tag['href']
            pdf_url = urljoin(BASE_URL, relative_pdf_url)
            link_text_str = link_tag.get_text(strip=True)
        else:
            # This warning will now trigger if no <a> tag is found directly in item_div
            print(f"Warning: Could not find a valid <a> PDF link directly inside a 'div.views-row'.")
            # We might still want to extract the date if the text is available,
            # but without a PDF link, the entry is not useful for downloading.
            # So, we can 'continue' to the next item_div if no link.
            # However, let's get the row_text_content first for debugging.
            row_text_content_for_debug = item_div.get_text(separator=' ', strip=True)
            print(f"Row text for skipped item: {row_text_content_for_debug[:150]}...")
            continue # Skip to the next views-row if no PDF link

        # 2. Extract Publication Date using regex on the text of the entire views-row
        row_text_content = item_div.get_text(separator=' ', strip=True)
        
        import re
        date_match = re.search(r"(\w+\s+\d{1,2},\s*\d{4})", row_text_content)
        if date_match:
            publication_date_str = date_match.group(1).replace(",", ", ")
            try:
                datetime.strptime(publication_date_str, "%B %d, %Y")
            except ValueError:
                print(f"Warning: Regex matched '{date_match.group(1)}' but failed strptime validation for {pdf_url}. Text: '{row_text_content[:150]}...'")
                publication_date_str = "Unknown"
        
        # This 'if pdf_url:' check is still good.
        if pdf_url:
            decision_links.append({
                'pdf_url': pdf_url,
                'publication_date': publication_date_str,
                'link_text': link_text_str
            })
        # The 'elif' for skipping is no longer needed here as we 'continue' above if no link_tag

    return decision_links

def download_pdf(pdf_url, publication_date, link_text):
    """Downloads a PDF and saves it with a structured name."""
    try:
        response = make_request(pdf_url) # This will also sleep for CRAWL_DELAY_SECONDS
        if not response:
            return False

        date_prefix = publication_date.replace(" ", "_").replace(",", "") if publication_date != "Unknown" else "UnknownDate"
        url_filename_part = os.path.basename(pdf_url).split('.')[0]
        safe_filename_part = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in url_filename_part[:50])
        filename = f"{date_prefix}_{safe_filename_part}.pdf"
        filepath = os.path.join(RAW_DATA_DIR, filename)

        if os.path.exists(filepath):
            print(f"File already exists: {filepath}. Skipping download.")
            return True

        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded: {filepath}")
        return True
    except Exception as e:
        print(f"Error downloading {pdf_url}: {e}")
        return False

# --- Main Script Logic ---
if __name__ == "__main__":
    print(f"Starting PDF acquisition for Topic ID: '{TARGET_URI_1}' ({TARGET_TOPIC_TEXT_DISPLAY}), Month ID: '{TARGET_M}' ({TARGET_MONTH_DISPLAY}), Year ID: '{TARGET_Y}' ({TARGET_YEAR_DISPLAY})")
    print(f"Output directory: {RAW_DATA_DIR}")
    print(f"User-Agent: {USER_AGENT}")
    print(f"Crawl Delay: {CRAWL_DELAY_SECONDS} seconds")

    all_found_decisions = []
    current_page_param = 0 
    items_per_page = 10

    while True:
        print(f"\n--- Fetching page (param value: {current_page_param}) ---")
        page_url = construct_filtered_url(TARGET_URI_1, TARGET_M, TARGET_Y, page=current_page_param, items_per_page=items_per_page)
        
        response = make_request(page_url)
        if not response:
            print(f"Failed to fetch page with param {current_page_param}. Stopping.")
            break
        
        # --- Optional: Save HTML of each fetched page for inspection if still having issues ---
        # debug_html_path = os.path.join(os.path.dirname(__file__), '..', f'debug_page_{current_page_param}.html')
        # with open(debug_html_path, 'w', encoding='utf-8') as f:
        #     f.write(response.text)
        # print(f"DEBUG: Saved HTML of page {current_page_param} to {debug_html_path}")
        # --- End Optional Save ---

        decisions_on_page = parse_decision_links(response.text)

        if not decisions_on_page:
            if current_page_param == 0: # No results on the first page
                 print(f"No decisions found by parser for the specified criteria on the first page: {page_url}")
            else: # No more results on subsequent pages
                print(f"No more decisions found by parser on page with param {current_page_param}. Reached end of results.")
            break
        
        print(f"Found {len(decisions_on_page)} decision links on page with param {current_page_param}.")
        all_found_decisions.extend(decisions_on_page)
        
        if len(decisions_on_page) < items_per_page:
            print("Fewer items than items_per_page, assuming last page.")
            break
            
        current_page_param += 1
        # Safety break for development, remove for production if expecting many pages
        if current_page_param > 5: # Increased safety break for testing pagination
            print("Safety break: Exceeded 5 pages (param value).")
            break

    print(f"\n--- Total decisions found: {len(all_found_decisions)} ---")
    if not all_found_decisions:
        print("No PDFs to download.")
    else:
        print("\n--- Starting PDF Downloads ---")
        download_count = 0
        for i, decision_info in enumerate(all_found_decisions):
            print(f"Downloading PDF {i+1}/{len(all_found_decisions)}: {decision_info['pdf_url']} (Date: {decision_info['publication_date']})")
            if download_pdf(decision_info['pdf_url'], decision_info['publication_date'], decision_info['link_text']):
                download_count += 1
        print(f"\n--- Download complete. Successfully downloaded {download_count}/{len(all_found_decisions)} PDFs. ---")

    print("Acquisition script finished.")