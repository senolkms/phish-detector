# utils/url_feats.py
import re
import numpy as np
import tldextract
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

class URLFeats(BaseEstimator, TransformerMixin):
    """
    Çıkış: [n_urls, n_ip_urls, n_suspicious_tld, n_unique_domains] (csr_matrix)
    """
    def __init__(self):
        self.suspicious_tlds = (".zip", ".top", ".xyz", ".ru", ".cn")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for txt in X:
            txt = txt if isinstance(txt, str) else str(txt)
            urls = re.findall(r"https?://\S+", txt, flags=re.IGNORECASE)

            n_urls = len(urls)
            n_ip = sum(bool(re.search(r"https?://\d{1,3}(?:\.\d{1,3}){3}", u)) for u in urls)
            n_susp = sum(u.lower().rstrip(".,);]").endswith(self.suspicious_tlds) for u in urls)

            domains = set()
            for u in urls[:50]:
                ext = tldextract.extract(u)
                dom = ".".join([p for p in [ext.domain, ext.suffix] if p])
                if dom:
                    domains.add(dom)
            n_dom = len(domains)

            rows.append([n_urls, n_ip, n_susp, n_dom])

        arr = np.asarray(rows, dtype=float)
        return sparse.csr_matrix(arr)
