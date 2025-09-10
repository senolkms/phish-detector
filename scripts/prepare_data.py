# scripts/prepare_data.py
import os, re, glob, sys, quopri, mailbox
from pathlib import Path
from email import policy
from email.parser import Parser
import email

import pandas as pd
from bs4 import BeautifulSoup

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# -------------------- Yardımcılar --------------------
def html_to_text(html: str) -> str:
    """HTML'i düz metne çevir (parser fallback: lxml -> html5lib -> html.parser)."""
    for parser in ("lxml", "html5lib", "html.parser"):
        try:
            return BeautifulSoup(html, parser).get_text(" ")
        except Exception:
            continue
    return re.sub(r"<[^>]+>", " ", html)

def msg_to_text(msg: email.message.Message) -> str:
    """email.message => subject + body (multipart, quoted-printable, html dönüştürme)."""
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = (part.get_content_type() or "").lower()
            if ctype in ("text/plain", "text/html"):
                try:
                    payload = part.get_payload(decode=True)
                    text = (
                        payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
                        if isinstance(payload, (bytes, bytearray))
                        else str(part.get_payload())
                    )
                except Exception:
                    text = str(part.get_payload())
                if "quoted-printable" in (part.get("Content-Transfer-Encoding", "").lower()):
                    try:
                        text = quopri.decodestring(text).decode("utf-8", errors="ignore")
                    except Exception:
                        pass
                if "html" in ctype or "<html" in text.lower():
                    text = html_to_text(text)
                parts.append(text)
    else:
        payload = msg.get_payload(decode=True)
        if payload is None:
            payload = msg.get_payload()
        if isinstance(payload, (bytes, bytearray)):
            try:
                text = payload.decode(msg.get_content_charset() or "utf-8", errors="ignore")
            except Exception:
                text = payload.decode("latin-1", errors="ignore")
        else:
            text = str(payload)
        if "quoted-printable" in (msg.get("Content-Transfer-Encoding", "").lower()):
            try:
                text = quopri.decodestring(text).decode("utf-8", errors="ignore")
            except Exception:
                pass
        if "<html" in text.lower():
            text = html_to_text(text)
        parts.append(text)

    subject = msg.get("subject") or ""
    out = (subject + " " + " ".join(parts)).strip()
    out = re.sub(r"\s+", " ", out)
    return out

def split_mboxish_text(raw: str) -> list[str]:
    """Nazario phishing-20xx.txt dosyalarını 'From ' satırlarıyla bloklara böl."""
    blocks = re.split(r'(?m)^(?=From .*\d{4})', raw)
    blocks = [b.strip() for b in blocks if b.strip()]
    return blocks

def read_eml(fp: Path) -> tuple[str, str]:
    """EML'den subject + body çıkar."""
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        msg = email.message_from_file(f)
    subject = msg.get("subject") or ""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = (part.get_content_type() or "").lower()
            if ctype in ("text/plain", "text/html"):
                try:
                    payload = part.get_payload(decode=True).decode(errors="ignore")
                except Exception:
                    payload = str(part.get_payload())
                if "html" in ctype or "<html" in payload.lower():
                    payload = html_to_text(payload)
                body += "\n" + payload
    else:
        payload = msg.get_payload(decode=True)
        if payload is None:
            payload = msg.get_payload()
            if isinstance(payload, list):
                payload = " ".join(map(str, payload))
        try:
            text = payload.decode(errors="ignore") if isinstance(payload, (bytes, bytearray)) else str(payload)
        except Exception:
            text = str(payload)
        if "<html" in text.lower():
            text = html_to_text(text)
        body = text
    return subject.strip(), body.strip()

def normalize_text(t: str) -> str:
    t = (t or "").replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -------------------- SpamAssassin --------------------
def collect_spamassassin() -> pd.DataFrame:
    rows = []
    base = RAW / "spamassassin"
    print(f"[INFO] scanning SpamAssassin: {base}", file=sys.stderr)
    for sub in ["easy_ham", "hard_ham"]:
        folder = base / sub
        print(f"  - {folder}", file=sys.stderr)
        for fp in glob.glob(str(folder / "*")):
            p = Path(fp)
            if not p.is_file():
                continue
            try:
                s, b = read_eml(p)
                rows.append({"text": normalize_text(s + " " + b), "label": 0, "source": f"spamassassin/{sub}"})
            except Exception as e:
                print(f"[WARN] skip {fp}: {e}", file=sys.stderr)

    spam_folder = base / "spam"
    if spam_folder.exists():
        print(f"  - {spam_folder}", file=sys.stderr)
        for fp in glob.glob(str(spam_folder / "*")):
            p = Path(fp)
            if not p.is_file():
                continue
            try:
                s, b = read_eml(p)
                rows.append({"text": normalize_text(s + " " + b), "label": 1, "source": "spamassassin/spam"})
            except Exception as e:
                print(f"[WARN] skip {fp}: {e}", file=sys.stderr)

    print(f"[INFO] spamassassin collected: {len(rows)}", file=sys.stderr)
    return pd.DataFrame(rows)

# -------------------- Nazario --------------------
def collect_nazario() -> pd.DataFrame:
    rows = []
    root = RAW / "nazario"
    print(f"[INFO] scanning Nazario: {root}", file=sys.stderr)

    for p in sorted(root.glob("*")):
        if not p.is_file():
            continue
        name = p.name.lower()
        try:
            if name.endswith(".mbox"):
                mb = mailbox.mbox(p, factory=lambda f: email.message_from_binary_file(f, policy=policy.default))
                kept = 0
                for msg in mb:
                    text = msg_to_text(msg)
                    if len(text) > 40:
                        rows.append({"text": text, "label": 1, "source": f"nazario:{p.name}"})
                        kept += 1
                print(f"  + {p.name}: mbox messages={kept}", file=sys.stderr)

            elif name.startswith("phishing-") and name.endswith(".txt"):
                raw = p.read_text(encoding="utf-8", errors="ignore")
                blocks = split_mboxish_text(raw)
                kept = 0
                for b in blocks:
                    msg = Parser(policy=policy.default).parsestr(b)
                    text = msg_to_text(msg)
                    if len(text) > 40:
                        rows.append({"text": text, "label": 1, "source": f"nazario:{p.name}"})
                        kept += 1
                print(f"  + {p.name}: blocks={len(blocks)} kept={kept}", file=sys.stderr)

            elif name.endswith(".eml"):
                s, b = read_eml(p)
                text = normalize_text(s + " " + b)
                if len(text) > 40:
                    rows.append({"text": text, "label": 1, "source": f"nazario:{p.name}"})

            else:
                raw = p.read_text(encoding="utf-8", errors="ignore")
                if "<html" in raw.lower():
                    raw = html_to_text(raw)
                text = normalize_text(raw)
                if len(text) > 40:
                    rows.append({"text": text, "label": 1, "source": f"nazario:{p.name}"})

        except Exception as e:
            print(f"[WARN] Nazario skip {p}: {e}", file=sys.stderr)

    print(f"[INFO] nazario collected: {len(rows)}", file=sys.stderr)
    return pd.DataFrame(rows)

# -------------------- Kaggle CSV'ler (genel şema tespitli) --------------------
def collect_kaggle_csvs() -> pd.DataFrame:
    """
    data/raw/kaggle altındaki tüm .csv dosyalarını okur ve
    text/label normalize eder. 7 dosyanın hepsini kapsayacak şekilde
    otomatik kolon tespiti yapar.
    """
    base = RAW / "kaggle"
    print(f"[INFO] scanning Kaggle: {base}", file=sys.stderr)
    all_rows = []

    # Kolon adayları
    text_cols  = ["text_combined", "text", "body", "message", "content", "email_text", "Message"]
    subj_cols  = ["subject", "Subject"]
    extra_cols = ["sender", "Sender", "receiver", "Receiver", "urls", "url", "date", "Date"]
    label_cols = ["label", "Label", "target", "class", "Category", "is_phishing", "phishing"]

    for fp in glob.glob(str(base / "*.csv")):
        p = Path(fp)
        try:
            df = pd.read_csv(p, encoding="utf-8", low_memory=False)
        except Exception:
            df = pd.read_csv(p, encoding="latin-1", low_memory=False)

        cols = list(df.columns)
        # ÖZEL: shantanudhakadd/spam.csv -> v1(label), v2(text)
        if set(["v1", "v2"]).issubset(cols):
            tmp = df[["v1", "v2"]].copy().rename(columns={"v1": "label", "v2": "text"})
            tmp["label"] = (
                tmp["label"].astype(str).str.lower().map({"spam": 1, "ham": 0})
            )
            tmp = tmp.dropna(subset=["label", "text"])
            tmp["label"] = tmp["label"].astype(int)
            tmp["source"] = f"kaggle/{p.name}"
            all_rows.append(tmp)
            print(f"  + loaded {p.name} (v1/v2) rows={len(tmp)}", file=sys.stderr)
            continue

        # Label kolonu
        label_col = next((c for c in cols if c in label_cols), None)
        if not label_col:
            print(f"[WARN] {p.name}: label column not found -> skip", file=sys.stderr)
            continue

        # Text oluşturma
        text_series = None

        # 1) hazır text benzeri bir kolon varsa
        for c in text_cols:
            if c in cols:
                text_series = df[c].fillna("").astype(str)
                break

        # 2) yoksa subject + body (+ opsiyonel meta) birleştir
        if text_series is None:
            subj = df[subj_cols[0]].fillna("").astype(str) if any(c in cols for c in subj_cols) else ""
            body = df["body"].fillna("").astype(str) if "body" in cols else ""
            extras = " ".join(df[c].fillna("").astype(str) for c in extra_cols if c in cols) if any(c in cols for c in extra_cols) else ""
            text_series = (subj + " " + body + " " + extras).astype(str)

        # Label normalizasyonu
        mapped = (
            df[label_col]
            .astype(str).str.strip().str.lower()
            .map({"spam":1, "ham":0, "phishing":1, "not phishing":0, "legit":0, "legitimate":0, "0":0, "1":1})
        )
        # mapping dışı sayıları da (örn 0/1 int) olduğu gibi al
        mapped = mapped.fillna(df[label_col])
        try:
            mapped = mapped.astype(int)
        except Exception:
            mapped = mapped.map(lambda x: 1 if str(x).strip().lower() in {"1","spam","phishing"} else 0)

        tmp = pd.DataFrame({
            "text": text_series.map(lambda s: re.sub(r"\s+", " ", str(s)).strip()),
            "label": mapped.astype(int, errors="ignore")
        }).dropna(subset=["text","label"])

        # çok kısa metinleri at
        tmp = tmp[tmp["text"].str.len() > 40]
        tmp["source"] = f"kaggle/{p.name}"
        all_rows.append(tmp)
        print(f"  + loaded {p.name} rows={len(tmp)}", file=sys.stderr)

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        print(f"[INFO] kaggle collected: {len(out)}", file=sys.stderr)
        return out

    print("[INFO] kaggle collected: 0", file=sys.stderr)
    return pd.DataFrame(columns=["text","label","source"])

# -------------------- Main --------------------
def main():
    parts = []
    if (RAW / "spamassassin").exists():
        parts.append(collect_spamassassin())
    if (RAW / "nazario").exists():
        parts.append(collect_nazario())
    if (RAW / "kaggle").exists():
        parts.append(collect_kaggle_csvs())

    if not parts:
        print("No raw data found. Please drop datasets under data/raw/...", file=sys.stderr)
        return

    df = pd.concat(parts, ignore_index=True)

    # Temizlik + dedup
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() > 40].dropna(subset=["text","label"])
    df = df.drop_duplicates(subset=["text"])

    print(df["label"].value_counts())
    out_fp = OUT / "emails_large_son.csv"
    df.to_csv(out_fp, index=False)
    print(f"Saved: {out_fp}  | rows={len(df)}")

if __name__ == "__main__":
    main()
