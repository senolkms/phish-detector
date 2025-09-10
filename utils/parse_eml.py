import email
from bs4 import BeautifulSoup

def parse_eml(file_path: str):
    """ .eml dosyasından subject ve body döndürür """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        msg = email.message_from_file(f)

    subject = msg["subject"] if msg["subject"] else ""
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                body += part.get_payload(decode=True).decode(errors="ignore")
            elif ctype == "text/html":
                html = part.get_payload(decode=True).decode(errors="ignore")
                body += BeautifulSoup(html, "lxml").get_text()
    else:
        body = msg.get_payload(decode=True).decode(errors="ignore")

    return subject.strip(), body.strip()
