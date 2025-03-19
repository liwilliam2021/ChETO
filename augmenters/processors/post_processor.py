import json
import re
import random

RAW_DATA_PATH = '../../data/text_output.json'
CHILEAN_DATA_PATH = '../../data/chilean_text.json'
FILTERED_DATA_PATH = '../../data/filtered_text.json'

"""
This script is used to merge the crawling data from multiple sources.
"""
def merge_crawling_data (paths):
    data = {}
    for path in paths:
        print (path)
        with open(path, 'r') as f:
            local_data = json.load(f)
        for key in local_data.keys():
            print (key, len (local_data[key]))
            if (len (local_data[key]) == 0):
                continue
            # Add new crawling job
            if key not in data:
                data[key] = local_data[key]
            # Override old crawling job with new ones
            elif len (local_data[key]) > len (data[key]):
                data[key] = local_data[key]
    print ("Merged data:")
    for key in data.keys():
        print (key, len (data[key]))
    with open(RAW_DATA_PATH, 'w') as f:
        json.dump(data, f)

"""
This script is used to filter out invalid paragraphs 
we suspect do not reflect natural language.
"""
def is_paragraph_valid (text):
    text = text.strip()
    if not text:
        return False

    # High numbers ratio
    num_ratio = sum(c.isdigit() for c in text) / len(text)
    if num_ratio > 0.2:
        return False

    # High uppercase ratio
    upper_ratio = sum(c.isupper() for c in text if c.isalpha()) / max(1, sum(c.isalpha() for c in text))
    if upper_ratio > 0.3:
        return False
  
    if re.search(r'(.)\1{5,}', text):  # same char ≥6 times
        return False

    space_ratio = text.count(' ') / len(text)
    # If there are barely any spaces, it's likely a title, header, or garbage like "________"
    if space_ratio < 0.05:
        return False
    # If too many spaces (e.g., due to bad HTML scraping like "&nbsp; &nbsp; &nbsp;"), also filter
    if space_ratio > 0.5:
        return False

    # Common footer/copyright/legalese/navigation/metadata terms (Chilean-focused too)
    if re.search(r'https?://|www\.|correo electrónico|email:|ISBN|©|todos los derechos reservados|'
                 r'política de privacidad|uso de cookies|mapa del sitio|bibliografía|'
                 r'referencias|fuente:|aviso legal|términos y condiciones|'
                 r'sitio web oficial|Ley de Protección de Datos|Ley N°|Decreto Supremo', text, re.IGNORECASE):
        return False

    # Check for high symbol ratio (junk like "|||" or "***")
    symbol_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
    if symbol_ratio > 0.4:
        return False

    # Too many line breaks (e.g., table or broken layout)
    word_count = len(text.split())
    if word_count < 10:
        return False
    line_breaks = text.count('\n')
    if line_breaks / word_count > (1 / 7):
        return False
    
    # If > 10% of characters fall outside the Latin blocks, flag as non-Spanish
    non_spanish_chars = sum(1 for c in text if not re.match(r'[\u0000-\u007F\u00C0-\u00FFáéíóúñÁÉÍÓÚÑüÜ]', c))
    ratio = non_spanish_chars / len(text)
    if ratio > 0.1:
        return False
    
    html_patterns = [
        r'&[a-zA-Z]+;',           # e.g., &nbsp;, &lt;, &amp;
        r'&#\d+;',                # e.g., &#160; (numeric entities)
        r'<\/?[a-z]+.*?>',        # e.g., <br>, </div>, <p>, <span style="...">
        r'<!--.*?-->',            # HTML comments
        r'class\s*=\s*["\'].*?["\']',  # HTML class attributes
        r'style\s*=\s*["\'].*?["\']',  # inline styles
        r'data-[a-zA-Z0-9\-]+\s*=\s*["\'].*?["\']'  # data-attributes like data-id=""
    ]
    for pattern in html_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            return False

    return True

"""
This script is used to filter out invalid paragraphs in post-processing step
"""
# It might be too aggressive, for now, TODO: keep conservative and refine
def post_process (path= RAW_DATA_PATH):
    with open(path, 'r') as f:
        data = json.load(f)
    bad_data = {}
    for key in data.keys():
        print (key)
        count = 0
        valid_data = []
        invalid_data = []
        for paragraph in data[key]:
            if is_paragraph_valid(paragraph):
                valid_data.append(paragraph)
            else:
                count += 1
                invalid_data.append(paragraph)
        print (f"Filtered {count} invalid paragraphs of {len(data[key])}")
        data[key] = valid_data
        bad_data[key] = invalid_data
    with open(CHILEAN_DATA_PATH, 'w', encoding="utf-8") as f:
        json.dump(data, f)
    with open(FILTERED_DATA_PATH, 'w', encoding="utf-8") as f:
        json.dump(bad_data, f)

# See how reasonable the filtered data looks
def print_filtered_data (num_samples=10):
    with open(FILTERED_DATA_PATH, 'r') as f:
        data = json.load(f)
    for key in data:
        print (key)
        # Sample 10 random paragraphs
        sampled_paragraphs = random.sample(data[key], min(num_samples, len(data[key])))

        for paragraph in sampled_paragraphs:
            print(paragraph)

if __name__ == '__main__':
    post_process()
    # print_filtered_data()