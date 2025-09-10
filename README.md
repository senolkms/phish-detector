# 🔎 Phishing E-mail Detection

Bu proje, **phishing (oltalama) e-postalarını tespit etmek** için geliştirilmiş bir makine öğrenmesi tabanlı sistemdir.  
Kullanıcılar `.eml` dosyalarını yükleyerek veya e-posta metnini yapıştırarak, sistemin e-postayı **güvenli mi yoksa phishing mi** olduğunu tahmin etmesini sağlayabilir.  

⚠ **Not:** Model yalnızca **İngilizce e-postalar** üzerinde optimize edilmiştir. Türkçe ve diğer diller için garanti verilmez.  

---

## 🔥 Özellikler
✅ **Phishing Tespiti:** TF-IDF + karakter n-gram + URL tabanlı özelliklerle sınıflandırma  
✅ **Streamlit Arayüzü:** `.eml` dosyası yükleme veya metin yapıştırma desteği  
✅ **Basit Kurallar Özeti:** E-postadaki URL sayısı, IP tabanlı linkler ve şüpheli TLD’ler  
✅ **Token Highlight:** Modelin en etkili bulduğu kelimeleri kırmızı (phishing yönlü) ve mavi (ham yönlü) renkle vurgulama  
✅ **Karar Eşiği Ayarı:** Kullanıcı slider ile kendi eşik değerini seçebilir  
✅ **Grafikler & Raporlar:** ROC, PR Curve, Confusion Matrix görselleri  

---

## 📌 Teknolojiler & Araçlar
| **Teknoloji** | **Açıklama** |
|--------------|-------------|
| Python | Genel backend ve model geliştirme |
| Scikit-learn | TF-IDF, SVC, Logistic Regression, pipeline |
| Pandas & NumPy | Veri işleme |
| BeautifulSoup | HTML e-posta temizleme |
| Streamlit | Kullanıcı arayüzü |
| Matplotlib | Görselleştirme |
| Joblib | Model kaydetme/yükleme |

---

## 📂 Proje Yapısı
```bash
PHISH-DETECTOR/
│
├── data/
│   ├── raw/                           # Ham veri kaynakları (Nazario, SpamAssassin, Kaggle, vb.)
│   └── processed/
│       └── emails_large_son.csv       # Birleştirilmiş ve temizlenmiş dataset
│
├── models/
│   └── phish_svc_tfidf_char_word_url_son.joblib   # (Google Drive üzerinden indirilmeli)
│
├── outputs/
│   ├── confusion_matrix.png           # Confusion matrix
│   ├── roc_curve.png                  # ROC curve
│   ├── pr_curve.png                   # Precision-Recall curve
│   └── metrics.json                   # Özet metrikler
│
├── scripts/
│   └── prepare_data.py                # Veri hazırlama scripti
│
├── utils/                             # Yardımcı modüller (parse_eml, text_clean, url_feats)
│
├── app.py                             # Streamlit uygulaması
├── train_large.py                     # Model eğitme scripti
└── README.md                          # Bu dosya

git clone https://github.com/senolkms/phish-detector.git
cd phish-detector
