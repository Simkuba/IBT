# Klasifikace síťového provozu pomocí contrastive learning
## Autor: Jakub Čoček (xcocek00@stud.fit.vutbr.cz)
## Vedoucí: Ing. Kamil Jeřábek, Ph.D.

---
### Instalace závislostí

Před spuštěním Jupyter notebooků a trénováním modelu je nutné nainstalovat všechny závislosti pro správné fungování. K tomu slouží soubor ```requirements.txt```. Instalaci je možné spustit následujícím příkazem:
```
pip install -r requirements.txt
```

Knihovna FlowMind, která je pro správné fungování modelu esenciální se ovšem v těchto závislostech nevyskytuje, jelikož je stále vyvíjena skupinou NES@FIT a není tedy veřejná.

---

### Struktura odevzdaných souborů

Odevzdané soubory jsou organizovány následovně:
- CESNET - Obsahuje model natrénovaný na CESNET datasetu a podsložky podle typu datové sady zvolené pro *fine-tuning*.
- common - Obsahuje soubory společné pro všechny datové sady jako je neuronová síť, dataloader a další.
- mirage19 - Obsahuje modely trénované na datové sadě Mirage19 a podsložky dle zvolené augmentace.
- mirage22 - Obsahuje modely trénované na datové sadě Mirage22 a podsložky dle zvolené augmentace.
- ucdavis - Obsahuje modely trénované na datové sadě Ucdavis-icdm19 a podsložky dle zvolené augmentace a testovací části datasetu.
- utmobilenet21 - Obsahuje modely trénované na datové sadě UTMobileNetTraffic2021 a podsložky dle zvolené augmentace.
- datasets.py - Skript pro analýzu datových sad.
- flowpics.py - Skript pro vytváření vizuálních FlowPic reprezentací.

Detailní popis obsahu obsahující většinu souborů:
```
├── CESNET
│   ├── human
│   │   └── fine-tuned_model.ipynb
│   ├── script
│   │   └── fine-tuned_model.ipynb
│   ├── mirage19
│   │   └── fine-tuned_model.ipynb
│   ├── mirage22
│   │   └── fine-tuned_model.ipynb
│   ├── utmobilenet21
│   │   └── fine-tuned_model.ipynb
│   └── model_rtt.ipynb
├── common
│   ├── augmentations.py
│   ├── dataloader.py
│   ├── dataset_analysis.py
│   ├── nn.py
│   └── tests.py
├── mirage19
│   ├── IAT
│   │   ├── filtered_mirage.ipynb
│   │   └── mirage.ipynb
│   ├── packet_loss
│   │   ├── filtered_mirage.ipynb
│   │   └── mirage.ipynb
│   ├── rotation
│   │   ├── filtered_mirage.ipynb
│   │   └── mirage.ipynb
│   └── RTT
│       ├── filtered_mirage.ipynb
│       └── mirage.ipynb
├── mirage22
│   ├── IAT
│   │   ├── mirage22-10.ipynb
│   │   └── mirage22-1000.ipynb
│   ├── packet_loss
│   │   ├── mirage22-10.ipynb
│   │   └── mirage22-1000.ipynb
│   ├── rotation
│   │   ├── mirage22-10.ipynb
│   │   └── mirage22-1000.ipynb
│   └── RTT
│       ├── mirage22-10.ipynb
│       └── mirage22-1000.ipynb
├── ucdavis
│   ├── IAT
│   │   ├── human
│   │   │   ├── 64ucdavis.ipynb
│   │   │   └── ucdavis.ipynb
│   │   └── script
│   │       ├── 64ucdavis.ipynb
│   │       └── ucdavis.ipynb
│   ├── packet_loss
│   │   ├── human
│   │   │   ├── 64ucdavis.ipynb
│   │   │   └── ucdavis.ipynb
│   │   └── script
│   │       ├── 64ucdavis.ipynb
│   │       └── ucdavis.ipynb
│   ├── rotation
│   │   ├── human
│   │   │   ├── 64ucdavis.ipynb
│   │   │   └── ucdavis.ipynb
│   │   └── script
│   │       ├── 64ucdavis.ipynb
│   │       └── ucdavis.ipynb
│   └── RTT
│       ├── human
│       │   ├── 64ucdavis.ipynb
│       │   └── ucdavis.ipynb
│       └── script
│           ├── 64ucdavis.ipynb
│           └── ucdavis.ipynb
├── utmobilenet21
│   ├── IAT
│   │   ├── filtered_utmobilenet21.ipynb
│   │   └── utmobilenet21.ipynb
│   ├── packet_loss
│   │   ├── filtered_utmobilenet21.ipynb
│   │   └── utmobilenet21.ipynb
│   ├── rotation
│   │   ├── filtered_utmobilenet21.ipynb
│   │   └── utmobilenet21.ipynb
│   └── RTT
│       ├── filtered_utmobilenet21.ipynb
│       └── utmobilenet21.ipynb
├── datasets.ipynb
├── flowpics.ipynb
├── requirements.txt
├── latex
│   ├── bib-styles
│   ├── obrazky-figures
│   ├── template-fig
│   ├── fitthesis.cls
│   ├── Makefile
│   ├── projekt-01-kapitoly-chapters.tex
│   ├── projekt-20-literatura-bibliography.bib
│   ├── projekt-30-prilohy-appendices.tex
│   ├── xcocek00.tex
│   ├── zadani.pdf
│   └── RTT
├── thesis
│   └── xcocek00-Klasifikace-sitoveho-provozu-pomoci-contrastive-learning.pdf
└── README.md
```