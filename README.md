# 🧠 BEYİN — 3B Nöral Simülasyon

Gerçek zamanlı, interaktif 3B beyin simülasyonu. Flask backend + Three.js frontend ile geliştirilmiştir.

---

## 🖥️ Özellikler

- **3B Beyin Modeli** — Gerçek GLB modeli, gyri/sulci (ak madde-boz madde) detayları ile
- **28 Beyin Bölgesi** — MNI koordinatlarıyla anatomik olarak doğru konumlandırılmış
- **Corpus Callosum** — Genu, Body, Splenium bölümleriyle
- **50 Beyin Modu** — Duygusal, bilişsel, klinik, fizyolojik ve uyku evreleri
- **Wilson-Cowan Sinir Ağı Dinamiği** — Bölge aktivasyonları nöral diferansiyel denklemlerle hesaplanır
- **Sinaptik Animasyonlar** — Bölgeler arası bağlantı akışları, tıklanınca bilgi sunar.
- **Damar Sistemi** — Arterler ve venler tıklanabilir, anatomik bilgi sunar
- **Nörotransmitter Veritabanı** — Dopamin, Serotonin, Noradrenalin, Glutamat, GABA, Asetilkolin. Sol tarafta sekmesi var üstüne tıklanınca scroll eylemek lazım.
- **60 Soruluk Nörodiverjans Anketi** — DSM-5 / ICD-11 / SCQ / GAD-7 / PHQ-9 temelli
- **Beyin Dalgası Görselleştirmesi** — Mod bazlı sinüzoidal dalgalar

---

## 🛠️ Kurulum

### Gereksinimler
```bash
pip install flask flask-cors numpy
```

### Çalıştırma
```bash
python aa.py
```
Tarayıcıda aç: `http://localhost:5002`

---
ÇALIŞMASI İÇİN İNDEX.HTML DOSYASINI DA İNDİRMEK GEREKİYOR. O DOSYA DA ANA KLASÖRÜN İÇİNDEKİ TEMPLATES KLASÖRÜNDE OLMALI!!!!!!
## 📁 Dosya Yapısı

```
aa/
├── aa.py              # Flask backend
├── brain.glb          # 3B beyin modeli
├── requirements.txt   # Python bağımlılıkları
└── templates/
    └── index.html     # Three.js frontend
```

---

## 🧬 Kullanılan Teknolojiler

| Teknoloji | Kullanım |
|-----------|----------|
| Python / Flask | Backend API |
| NumPy | Wilson-Cowan nöral dinamik hesaplama |
| Three.js | 3B görselleştirme |
| GLTFLoader | Beyin modeli yükleme |
| CSS2DRenderer | Bölge etiketleri |
| OrbitControls | Kamera kontrolü |

---

## 🧠 Nöral Dinamik Modeli

Bölge aktivasyonları **Wilson-Cowan denklemi** ile hesaplanır:

$$\tau \frac{dA_i}{dt} = -A_i + F\left(\sum_j W_{ij} A_j\right)$$

Burada $F(x) = \frac{1}{1+e^{-x}}$ sigmoid aktivasyon fonksiyonu, $W_{ij}$ ise bölgeler arası bağlantı ağırlıklarıdır.

---

## ⚠️ Yasal Uyarı

Bu uygulama yalnızca eğitim ve araştırma amaçlıdır. Nörodiverjans anketi sonuçları klinik tanı niteliği taşımaz. Profesyonel değerlendirme için bir uzmana başvurunuz.

---

## 📄 Lisans

MIT License
