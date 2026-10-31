# --- LIBRARIES --- #

import re
from nltk.stem.snowball import SnowballStemmer

# --- FUNCTIONS --- #

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower() #  Lowercase
    text = re.sub(r'(\d)\s*x\s*(\d)', r'\1x\2', text) #  Normalize dimensions: '8 x 40' -> '8x40'
    text = re.sub(r'(x\d+(?:[.,]\d+)?)\s*mm\b', r'\1', text) #  Strip 'mm' directly following a dimension: 'm8x40 mm' -> 'm8x40'
    
    return ' '.join(text.split()) #  Normalize multiple spaces to a single space


stemmer = SnowballStemmer("finnish")

def tokenize_stem(text):
    tokens = re.findall(r'\b\w+\b', str(text).lower())
    return [stemmer.stem(t) for t in tokens]

# --- TESTS --- #

if __name__ == "__main__":
    #tests
    print(clean_text("Kuusiokoloruuvi M8X40 A4"))
    print(clean_text("kuusiokoloruuvi m8 x 40 a4"))
    print(clean_text("kuusiokoloruuvi M8 x 40 mm A4"))
