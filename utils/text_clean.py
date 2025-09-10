import re

def clean_text(text: str):
    """ Basit temizlik: lower, url normalize, fazla boşluk temizle """
    text = text.lower()
    text = re.sub(r"http\S+", " [url] ", text)  # URL'leri işaretle
    text = re.sub(r"\s+", " ", text)  # fazla boşluk
    return text.strip()
