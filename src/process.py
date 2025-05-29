# src/process.py

import os
import json
import pdfplumber
import re
from datetime import datetime

# --- Configuration ---
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- Helper Functions ---

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file using pdfplumber."""
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n" # Add a newline between pages
                # else:
                #     print(f"Warning: No text found on page {i+1} of {pdf_path}")
        return full_text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def clean_text(text):
    """Basic text cleaning."""
    if not text:
        return ""
    text = text.replace('\r\n', '\n').replace('\r', '\n') # Normalize newlines
    text = re.sub(r'\s+\n', '\n', text) # Remove spaces before newlines
    text = re.sub(r'\n\s+', '\n', text) # Remove spaces after newlines
    text = re.sub(r'\n{3,}', '\n\n', text) # Reduce multiple newlines to max two
    text = text.strip()
    # Add any other specific cleaning rules as needed (e.g., removing headers/footers if identifiable)
    return text

def extract_metadata_from_text(text, filename=""):
    """
    Extracts metadata like case name, decision date, and headings from the text.
    This is highly heuristic and will need refinement based on document structure.
    """
    metadata = {
        "case_name": "Unknown",
        "decision_date_from_doc": "Unknown", # Date extracted from document text
        "headings": [] # List of {"text": "heading_text", "start_char_offset": offset}
    }
    if not text:
        return metadata

    # 1. Extract Decision Date (look for dates in the first few lines or common formats)
    #    Example: "Date: February 26, 2025" or just "February 26, 2025"
    #    We'll search in the first ~500 characters for a date.
    first_chunk_for_date = text[:500]
    date_pattern = r"(?:Date:\s*)?([A-Za-z]{3,9}\.?\s+\d{1,2},\s*\d{4})" # Month name/abbrev (optional period) Day, Year
    date_match = re.search(date_pattern, first_chunk_for_date, re.IGNORECASE)
    if date_match:
        try:
            date_str_from_doc = date_match.group(1).replace(".", "") # Remove period from month if present
            # Normalize to full month name if abbreviated, though strptime might handle some abbrevs
            # This part is tricky if we don't know all abbrevs. Python's strptime with %b handles standard 3-letter abbrevs.
            # Let's try parsing directly first.
            dt_obj = datetime.strptime(date_str_from_doc.replace(",", ", "), "%B %d, %Y") # Try full month name
        except ValueError:
            try: # Try with abbreviated month format %b
                dt_obj = datetime.strptime(date_str_from_doc.replace(",", ", "), "%b %d, %Y")
            except ValueError:
                print(f"Warning: Found potential date '{date_match.group(1)}' in {filename} but failed strptime with %B and %b.")
                metadata["decision_date_from_doc"] = "Unknown" # Keep as unknown
            else:
                metadata["decision_date_from_doc"] = dt_obj.strftime("%Y-%m-%d")
        else:
             metadata["decision_date_from_doc"] = dt_obj.strftime("%Y-%m-%d")

    # 2. Extract Case Name (heuristic: look for "Matter of" or "IN RE:")
    #    Often in the first few lines of the document.
    first_chunk_for_case = text[:1000] # Search in a larger chunk for case name
    case_name_match = re.search(r"(Matter of\s+[\w\s,-]+?)(?=\n|FILE:|Date:)", first_chunk_for_case, re.IGNORECASE)
    if not case_name_match: # Try another common pattern
        case_name_match = re.search(r"(IN RE:\s*[\w\s,-]+?)(?=\n|FILE:|Date:)", first_chunk_for_case, re.IGNORECASE)
    
    if case_name_match:
        # Clean up the matched case name
        case_name = case_name_match.group(1).strip()
        case_name = re.sub(r'\s+', ' ', case_name) # Normalize whitespace
        # Remove trailing commas or hyphens if any
        case_name = re.sub(r'[,\-\s]+$', '', case_name)
        metadata["case_name"] = case_name
    else:
        print(f"Warning: Could not extract case name for {filename} using 'Matter of' or 'IN RE:'.")


    # 3. Extract Headings (heuristic: look for ALL CAPS lines, or lines with specific keywords)
    #    This is very basic and will likely need significant improvement.
    lines = text.split('\n')
    current_offset = 0
    for line in lines:
        stripped_line = line.strip()
        # Heuristic 1: ALL CAPS line, relatively short, not ending with a period.
        if stripped_line.isupper() and 2 < len(stripped_line) < 100 and not stripped_line.endswith('.'):
            # Avoid lines that are just "I" or "I I" etc.
            if not all(c == 'I' or c.isspace() for c in stripped_line):
                if len(stripped_line.split()) < 7 : 
                    heading_keywords = ["DISCUSSION", "ANALYSIS", "FINDINGS", "CONCLUSION", "ORDER", "FACTS", "LAW", "ARGUMENT", "ISSUE", "BENEFIT", "DETERMINATION", "MEMBERSHIPS", "MATERIALS", "ARTICLES", "ROLES", "AWARDS"] # Expanded keywords
                    # Require keywords OR be very short (like a single word "LAW")
                    if any(keyword in stripped_line for keyword in heading_keywords) or len(stripped_line.split()) <= 2:
                        metadata["headings"].append({
                            "text": stripped_line,
                            "start_char_offset": current_offset + text[current_offset:].find(line)
                        })
        
        # Heuristic 2: Roman numerals/letters (this one seems good, keep as is or slightly refine word count)
        roman_match = re.match(r"^(?:[IVXLCDM]+\.|[A-Z]\.)\s+([\w\s,-]+)", stripped_line) # Allow comma in title
        if roman_match and len(roman_match.group(1).strip().split()) < 12 : # Slightly increased word count limit
             metadata["headings"].append({
                "text": stripped_line, 
                "start_char_offset": current_offset + text[current_offset:].find(line)
            })

        current_offset += len(line) + 1 # +1 for the newline character

    # Remove duplicate headings if any (based on text and nearby offset)
    unique_headings = []
    seen_headings_text = set()
    for h in sorted(metadata["headings"], key=lambda x: x["start_char_offset"]): # Sort by offset
        # A simple way to avoid near-duplicate headings if text is same and offset is close
        is_duplicate = False
        for uh in unique_headings:
            if uh["text"] == h["text"] and abs(uh["start_char_offset"] - h["start_char_offset"]) < 50: # 50 char tolerance
                is_duplicate = True
                break
        if not is_duplicate:
            unique_headings.append(h)
            seen_headings_text.add(h["text"]) # Add to seen set for exact text match later
        elif h["text"] not in seen_headings_text: # If text is different, it's not a duplicate by this logic
             unique_headings.append(h)
             seen_headings_text.add(h["text"])


    metadata["headings"] = unique_headings

    return metadata

def create_json_structure(pdf_filename, raw_text_path, cleaned_text, extracted_metadata, acquisition_details=None):
    """
    Creates the JSON structure for the processed document.
    'acquisition_details' could be a dict passed from acquire.py if we had more info per file,
    e.g., exact publication date from website, source_url of the PDF.
    For now, we'll derive some from filename if needed.
    """
    # Try to get original publication date from filename (as downloaded by acquire.py)
    # Filename format: February_26__2025_FEB262025_02B2203.pdf
    publication_date_from_filename = "Unknown"
    try:
        date_part_fn = pdf_filename.split('_')[0] + " " + pdf_filename.split('_')[1] + ", " + pdf_filename.split('_')[2]
        dt_obj = datetime.strptime(date_part_fn.replace("__", " "), "%B %d, %Y") # Handles double underscore if present
        publication_date_from_filename = dt_obj.strftime("%Y-%m-%d")
    except Exception: # Broad except as filename format might vary or be "UnknownDate"
        pass
        
    # Construct source_url (this is a guess based on common patterns, might not be the exact one from acquire.py)
    # Ideally, acquire.py would pass this along if we were processing one by one.
    # For now, this is a placeholder.
    placeholder_source_url = f"https://www.uscis.gov/sites/default/files/err/.../{pdf_filename}" # Simplified

    doc_id = pdf_filename.replace(".pdf", "") # Simple ID from filename

    data = {
        "document_id": doc_id,
        "source_url": acquisition_details.get("pdf_url", placeholder_source_url) if acquisition_details else placeholder_source_url,
        "retrieval_date": datetime.now().isoformat(), # When this processing script ran
        "publication_date_on_website": acquisition_details.get("publication_date", publication_date_from_filename) if acquisition_details else publication_date_from_filename,
        "case_name": extracted_metadata["case_name"],
        "decision_date_from_doc": extracted_metadata["decision_date_from_doc"],
        "decision_type": "I-140 Extraordinary Ability", # Known from acquisition target
        "document_format": "pdf",
        "raw_text_path": raw_text_path, # Path to the original downloaded PDF
        "cleaned_text": cleaned_text,
        "headings": extracted_metadata["headings"],
        "processing_metadata": {
            "text_extraction_tool": f"pdfplumber v{pdfplumber.__version__}",
            "metadata_extraction_heuristics_version": "1.0" # Version your heuristics
        }
    }
    return data

# --- Main Script Logic ---
if __name__ == "__main__":
    print(f"Starting processing of PDFs from: {RAW_DATA_DIR}")
    print(f"Outputting JSON to: {PROCESSED_DATA_DIR}")

    pdf_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in the raw data directory.")
    else:
        print(f"Found {len(pdf_files)} PDF files to process.")

    processed_count = 0
    for i, pdf_filename in enumerate(pdf_files):
        print(f"\n--- Processing file {i+1}/{len(pdf_files)}: {pdf_filename} ---")
        pdf_path = os.path.join(RAW_DATA_DIR, pdf_filename)

        # In a more integrated pipeline, acquire.py might generate a manifest file
        # with details like the exact source_url and website_publication_date for each PDF.
        # For now, we don't have that, so create_json_structure will use placeholders or derive.
        # Example: acquisition_details = load_manifest_for(pdf_filename) 
        acquisition_details_placeholder = { # Simulate what acquire.py might provide
            "pdf_url": f"https://www.uscis.gov/some_path/{pdf_filename}", # This is a guess
            "publication_date": "Unknown" # We'll try to get this from filename in create_json_structure
        }
        # If acquire.py saved a manifest with original PDF URLs and publication dates, load it here.
        # For now, we'll use the filename to try and get the publication date.
        # The filename format from acquire.py was like: February_26__2025_FEB262025_02B2203.pdf
        try:
            fn_parts = pdf_filename.split('_')
            # Month_Day__Year_...
            # fn_parts[0] = Month (e.g., "February")
            # fn_parts[1] = Day (e.g., "03" or "3")
            # fn_parts[2] = Year (e.g., "2025")
            # We need to handle the double underscore between Day and Year carefully if it's part of the split.
            # Example: "February_03__2025_..." -> parts: ["February", "03", "", "2025", ...]
            # Example: "February_3__2025_..." -> parts: ["February", "3", "", "2025", ...]
        
            month_str = fn_parts[0]
            day_str = fn_parts[1]
            year_str = ""
        
            # Find the year part, which should be 4 digits and after the day
            # The filename structure is Month_DD__YYYY_...
            # So, after splitting by '_', if fn_parts[2] is empty (due to "__"), then fn_parts[3] is the year.
            # Otherwise, fn_parts[2] is the year.
            if fn_parts[2] == "" and len(fn_parts) > 3 and fn_parts[3].isdigit() and len(fn_parts[3]) == 4:
                year_str = fn_parts[3]
            elif fn_parts[2].isdigit() and len(fn_parts[2]) == 4: # Handles cases like Month_DD_YYYY_...
                year_str = fn_parts[2]
            
            if month_str and day_str and year_str:
                parsed_date_from_fn = datetime.strptime(f"{month_str} {day_str}, {year_str}", "%B %d, %Y")
                acquisition_details_placeholder["publication_date"] = parsed_date_from_fn.strftime("%Y-%m-%d")
            else:
                print(f"Warning: Could not reliably parse date components from filename '{pdf_filename}'")
        except Exception as e_fn_date:
            print(f"Warning: Error parsing date from filename '{pdf_filename}': {e_fn_date}")
            pass # Keep as "Unknown"


        raw_text = extract_text_from_pdf(pdf_path)
        if raw_text is None:
            print(f"Skipping {pdf_filename} due to text extraction error.")
            continue

        cleaned_text = clean_text(raw_text)
        
        # Pass filename to metadata extraction for better warning messages
        extracted_metadata = extract_metadata_from_text(cleaned_text, pdf_filename) 
        
        json_data = create_json_structure(pdf_filename, pdf_path, cleaned_text, extracted_metadata, acquisition_details_placeholder)
        
        json_filename = pdf_filename.replace(".pdf", ".json")
        json_filepath = os.path.join(PROCESSED_DATA_DIR, json_filename)
        
        try:
            with open(json_filepath, 'w', encoding='utf-8') as jf:
                json.dump(json_data, jf, indent=4, ensure_ascii=False)
            print(f"Successfully processed and saved: {json_filepath}")
            processed_count += 1
        except Exception as e:
            print(f"Error saving JSON for {pdf_filename}: {e}")

    print(f"\n--- Processing complete. Successfully processed {processed_count}/{len(pdf_files)} PDFs. ---")