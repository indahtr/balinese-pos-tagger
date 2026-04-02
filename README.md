# Balinese POS Tagger (Bi-LSTM)

Library Python untuk melakukan **Part-of-Speech (POS) Tagging** khusus Bahasa Bali menggunakan arsitektur **Bidirectional Long Short-Term Memory (Bi-LSTM)**. Library ini dirancang untuk mendukung penelitian NLP pada bahasa daerah (*low-resource language*).

> **Performa Model:** Model ini dilatih menggunakan dataset yang disusun dan dianotasi secara manual (*hand-annotated*) khusus untuk penelitian ini. Berdasarkan hasil pengujian, arsitektur Bi-LSTM mencapai tingkat **Akurasi 88,11%** dan **F1-Score 86,11%**.

## Instalasi
Library dapat diinstal secara langsung melalui repositori GitHub menggunakan perintah :
```bash
pip install git+https://github.com/indahtr/balinese-pos-tagger.git
```
## Cara Penggunaan
Untuk pengujian awal, jalankan skrip 'demo.py' yang tersedia pada direktori utama dengan cara:
```bash
python demo.py
```

### Penggunaan Dasar
```python
from balinese_pos_tagger import PosTagger

tagger = PosTagger()
text = "Mangkin kantun masisa malih asiki sané ngranayang Baliné kasub, inggih punika adat utawi budayané."
result = tagger.tag(text)

print(result)
```
### Contoh Output
```bash
[('Mangkin', 'RB'), ('kantun', 'RB'), ('masisa', 'VB'), ('malih', 'RB'), ('asiki', 'CD'), ('sane', 'PR'), ('ngranayang', 'VB'), ('Baline', 'NNP'), ('kasub', 'JJ'), (',', 'Z'), ('inggih', 'VB'), ('punika', 'DT'), ('adat', 'NN'), ('utawi', 'CC'), ('budayane', 'NN'), ('.', 'Z')]
```

## Dependensi
- `python >= 3.9`
- `torch >= 2.0.0` (PyTorch)
- `numpy >= 1.23.0`

## Atribut dan Metadata Model
Konfigurasi arsitektur dan metadata model tersedia melalui atribut objek untuk kebutuhan validasi maupun analisis penelitian:
```python
print(f"Versi Library       : {tagger.__version__}")
print(f"Dimensi Embedding   : {tagger.embedding_dim}")
print(f"Hidden Layer Size   : {tagger.hidden_size}")
print(f"Max Sequence Length : {tagger.max_len}")
```

## Daftar Tagset
Berikut adalah 19 kategori tagset yang digunakan dalam model ini:

| **No** | **Tag** | **Deskripsi**                               | **Contoh**                                |
|--------|---------|---------------------------------------------|-------------------------------------------|
| 1      | CC      | Coordinating conjunction/coordinator        | lan, tur, sakéwala, nanging               |
| 2      | CD      | Cardinal number                             | abesik, séket, kalih, 1973                |
| 3      | DT      | Determiner / article                        | niki, punika, Mén, Ni                     |
| 4      | FW      | Foreign word                                | online, handphone                         |
| 5      | IN      | Preposition                                 | di, ka, uli, ring                         |
| 6      | JJ      | Adjective                                   | ageng, akéh, selem, tua                   |
| 7      | MD      | Modal and auxiliary verb                    | sampun, patut, pacang, dados,             |
| 8      | NEG     | Negation                                    | sing, nénten, tusing, durung              |
| 9      | NN      | Noun                                        | jalér, toko, budaya, krama                |
| 10     | NND     | Classifier, partitive, and measurement noun | ukud, katih, abulih, abungkul             |
| 11     | NNP     | Proper noun                                 | Bali, Indonesia, Nyepi, Nyoman            |
| 12     | PR      | Pronoun                                     | tiang, sané, ento, ipun                   |
| 13     | RB      | Adverb                                      | kapah, tuni, adéng-adéng, enggal-enggal,  |
| 14     | RP      | Particle                                    | ja, ya, téh, ké                           |
| 15     | SC      | Subordinating conjunction/subordinator      | sawiréh, saking, santukan, dugas          |
| 16     | UH      | Interjection                                | jeg, pih, arah, yeh                       |
| 17     | VB      | Verb                                        | meli, rauh, nepukin, nuturang             |
| 18     | WH      | Question                                    | sira, napi, nyén, dija                   |
| 19     | Z       | Punctuation                                 | . , ! ? : ; ( ) “ ” -                     |

## Struktur Proyek
```text
balinese-pos-tagger/
├── balinese_pos_tagger/              # Main Python package
│   ├── resources/                    # Model artifacts & vocabulary mappings
│   │   ├── model.pt                  # Trained BiLSTM model weights
│   │   ├── params.json               # Model hyperparameters configuration
│   │   ├── word2idx.json             # Token-to-index mapping
│   │   └── idx2tag.json              # Index-to-tag mapping
│   │
│   ├── utils/                        # Utility modules for text preprocessing
│   │   └── text_utils.py           
│   │
│   ├── __init__.py                   # Package initialization & public API exposure
│   ├── io.py                         # Artifact loading (model, config, vocab)
│   ├── model.py                      # BiLSTM model architecture (PyTorch)
│   └── tagger.py                     # Core POS tagging pipeline (inference logic)
│
├── demo.py                           # Example script demonstrating basic usage
├── pyproject.toml                    # Build system and dependency configuration
└── README.md                         # Project documentation
```

## License
MIT License