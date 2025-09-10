# app.py
import re
import joblib
import streamlit as st
import tldextract
import numpy as np
from html import escape

from utils.parse_eml import parse_eml
from utils.text_clean import clean_text
from utils.url_feats import URLFeats  # pipeline içindeki URLFeats için

# Eski pickle'larla uyum
URLFeats.__module__ = "__main__"

# ---------- App config ----------
st.set_page_config(page_title="Phishing E-mail Detection", page_icon="🔎")
st.title("🔎 Phishing E-mail Detection")

# ---------- Modeli yükle ----------
@st.cache_resource
def load_pipe():
    return joblib.load("models/phish_svc_tfidf_char_word_url_son.joblib")

pipe = load_pipe()

# ---------- FeatureUnion’dan bileşenleri al ----------
FEAT = pipe.named_steps.get("features", None)
WORD_VEC = CHAR_VEC = URL_BLOCK = None
if FEAT is not None and hasattr(FEAT, "transformer_list"):
    for name, tr in FEAT.transformer_list:
        if name == "word":
            WORD_VEC = tr
        elif name == "char":
            CHAR_VEC = tr
        elif name == "url":
            URL_BLOCK = tr

# (Opsiyonel) LogisticRegression’lı eski pipeline desteği
LR_MODE, LR_VEC, LR_CLF = False, None, None
try:
    LR_VEC = pipe.named_steps["tfidfvectorizer"]
    LR_CLF = pipe.named_steps["logisticregression"]
    LR_MODE = True
except Exception:
    LR_MODE = False

# ---------- Girdi ----------
uploaded_file = st.file_uploader("Bir .eml dosyası yükleyin", type=["eml"])
input_text = st.text_area("Ya da e-posta metnini buraya yapıştırın:")

text = None
if uploaded_file:
    with open("temp.eml", "wb") as f:
        f.write(uploaded_file.read())
    subject, body = parse_eml("temp.eml")
    text = (subject or "") + " " + (body or "")
elif input_text:
    text = input_text

# ---------- Yardımcılar ----------
HEADER_PREFIXES = (
    "date:", "message-id:", "to:", "from:", "reply-to:", "mime-version:",
    "content-type:", "content-transfer-encoding:", "x-", "received:",
    "return-path:", "authentication-results:", "arc-", "dkim-", "dmarc:", "spf:"
)

def strip_headers_for_tokens(raw_text: str) -> str:
    """Token etkilerini çıkarırken header satırlarını görmezden gel."""
    head, sep, body = raw_text.partition("\n\n")
    txt = body if sep else raw_text
    cleaned = []
    for line in txt.splitlines():
        if any(line.lower().startswith(p) for p in HEADER_PREFIXES):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

# Masum / çok genel kelimeler (küçük harf)
SAFE_EXCLUDE = {
    "style","http","https","www","com","org","net","co","src","span",
    "welcome","hello","hi","thanks","regards","team","please","dear","kind",
    "imagine","idea","ideas","unique","best","today","tomorrow","partner",
    "thinking","thought","contact","help","support","update","information"
}

def _filter_tokens(pairs, top_k=5):
    """Görsel/HTML artıklarını ve çok kısa/karışık tokenları ele."""
    out = []
    for w, v in pairs:
        w2 = w.strip().lower()
        if w2 in SAFE_EXCLUDE:
            continue
        # yalnızca alfabetik, en az 4 harf
        if not re.fullmatch(r"[a-zğüşöçıİĞÜŞÖÇ]{4,20}", w2):
            continue
        out.append((w, v))
        if len(out) >= top_k:
            break
    return out

def highlight_text(raw_text: str, pos_tokens, neg_tokens):
    """Önce HTML-escape et; sonra sadece gerçek içerikte highlight yap."""
    colored = escape(raw_text)

    pos_tokens = sorted(pos_tokens, key=lambda x: len(x[0]), reverse=True)
    neg_tokens = sorted(neg_tokens, key=lambda x: len(x[0]), reverse=True)

    def wrap(text, word, color):
        # escape edilmiş metinde kelime sınırında ara
        pattern = re.compile(rf"(?i)\b{re.escape(word)}\b")
        return pattern.sub(
            lambda m: f"<span style='background:{color};padding:2px 4px;border-radius:4px'>{m.group(0)}</span>",
            text,
        )

    for w, _ in pos_tokens[:5]:
        colored = wrap(colored, w, "#ffcccc")   # phishing (kırmızımsı)
    for w, _ in neg_tokens[:5]:
        colored = wrap(colored, w, "#cce5ff")   # ham (mavimsi)
    return colored

def simple_rules_stats(raw_text: str) -> dict:
    urls = re.findall(r"https?://\S+", raw_text, flags=re.IGNORECASE)
    ip_links = [u for u in urls if re.search(r"https?://\d{1,3}(?:\.\d{1,3}){3}", u)]
    suspicious_tlds = (".zip", ".top", ".xyz", ".ru", ".cn")
    sus_tld_hits = [u for u in urls if u.lower().rstrip(".,);]").endswith(suspicious_tlds)]
    domains = []
    for u in urls[:10]:
        ext = tldextract.extract(u)
        dom = ".".join([p for p in [ext.domain, ext.suffix] if p])
        if dom:
            domains.append(dom)
    return {
        "url_count": len(urls),
        "ip_url_count": len(ip_links),
        "suspicious_tld_count": len(sus_tld_hits),
        "urls": urls[:10],
        "domains": domains,
    }

# ---------- Token etkileri ----------
def lr_top_tokens(doc_text: str, top_k: int = 5):
    X = LR_VEC.transform([doc_text])
    coefs = LR_CLF.coef_.ravel()
    idx = X.nonzero()[1]
    contrib = {LR_VEC.get_feature_names_out()[i]: float(X[0, i] * coefs[i]) for i in idx}
    pos = sorted([(t, v) for t, v in contrib.items() if v > 0], key=lambda x: x[1], reverse=True)
    neg = sorted([(t, v) for t, v in contrib.items() if v < 0], key=lambda x: x[1])
    return _filter_tokens(pos, top_k), _filter_tokens(neg, top_k)

def svc_top_tokens(doc_text: str, top_k: int = 5):
    """
    Calibrated LinearSVC katsayılarını word TF-IDF dilimi üzerinde kullan.
    Dinamik eşik: max(|katkı|)’nın %3’ü ve en az 0.002.
    """
    try:
        clf = pipe.named_steps["clf"]

        coef = None
        if hasattr(clf, "calibrated_classifiers_"):
            for cc in clf.calibrated_classifiers_:
                est = getattr(cc, "estimator", None)
                if hasattr(est, "coef_"):
                    coef = est.coef_.ravel()
                    break
        elif hasattr(clf, "base_estimator") and hasattr(clf.base_estimator, "coef_"):
            coef = clf.base_estimator.coef_.ravel()

        if coef is None or WORD_VEC is None or CHAR_VEC is None:
            return [], []

        n_word = len(WORD_VEC.get_feature_names_out())
        word_coef = coef[:n_word]

        Xw = WORD_VEC.transform([doc_text])                  # 1 x n_word (sparse)
        contrib_vals = (Xw.multiply(word_coef)).toarray()[0] # katkı dizisi
        terms = WORD_VEC.get_feature_names_out()

        # Dinamik eşik
        max_abs = float(np.max(np.abs(contrib_vals))) if contrib_vals.size else 0.0
        thr = max(0.03 * max_abs, 0.002)

        order_pos = np.argsort(-contrib_vals)
        order_neg = np.argsort(contrib_vals)

        pos_all = [(terms[i], float(contrib_vals[i])) for i in order_pos if contrib_vals[i] >  thr]
        neg_all = [(terms[i], float(contrib_vals[i])) for i in order_neg if contrib_vals[i] < -thr]

        return _filter_tokens(pos_all, top_k), _filter_tokens(neg_all, top_k)
    except Exception:
        return [], []

# ---------- Inference ----------
if text:
    # Tahmin için temiz metin
    clean_for_pred = clean_text(text)
    proba = float(pipe.predict_proba([clean_for_pred])[0][1])

    st.subheader("📊 Tahmin")
    th = st.slider("Karar eşiği", 0.05, 0.95, 0.60, 0.01)  # default 0.60
    pred = 1 if proba >= th else 0

    if pred == 1:
        st.error(f"⚠️ Phishing (p={proba:.2f}) — eşik={th:.2f}")
    else:
        st.success(f"✅ Güvenli (p={proba:.2f}) — eşik={th:.2f}")

    # Basit kurallar
    stats = simple_rules_stats(text)
    with st.expander("🔎 Basit Kurallar Özeti"):
        st.write(
            f"- URL sayısı: **{stats['url_count']}**  | IP URL: **{stats['ip_url_count']}**  | Şüpheli TLD: **{stats['suspicious_tld_count']}**"
        )
        if stats["domains"]:
            st.write("Alan adları:", ", ".join(set(stats["domains"])))
        if stats["urls"]:
            st.write("Örnek URL'ler:")
            for u in stats["urls"]:
                st.write(f"- {u}")

    # Token etkileri: header’sız metin üzerinde çıkar
    token_text = strip_headers_for_tokens(text)

    if LR_MODE and LR_VEC is not None and LR_CLF is not None:
        pos, neg = lr_top_tokens(token_text, top_k=5)
        expander_title = "🧠 En Etkili Kelimeler (LR katsayı × tf-idf)"
    else:
        pos, neg = svc_top_tokens(token_text, top_k=5)
        expander_title = "🧠 En Etkili Kelimeler (SVC katsayı × tf-idf)"

    with st.expander(expander_title):
        if pos or neg:
            st.write("**Phishing yönlü (+):**", [w for w, _ in pos])
            st.write("**Ham yönlü (−):**", [w for w, _ in neg])
        else:
            st.info("Bu modelden token bazlı etki listesi çıkarılamadı.")

    # Highlight
    colored_html = highlight_text(text, pos, neg)
    st.markdown("**Örnek İçerik (highlight edilmiş):**", unsafe_allow_html=True)
    st.markdown(f"<div style='white-space: pre-wrap'>{colored_html}</div>", unsafe_allow_html=True)

else:
    st.info("Bir .eml dosyası yükleyin veya metni yapıştırın.")
