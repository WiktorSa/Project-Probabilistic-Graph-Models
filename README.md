# Project-Probabilistic-Graph-Models

Whole project is in Polish language.

# Projekt Probabilistyczne Modele Grafowe - Predykcja poziomów otyłości

AA | Wiktor Sadowy | Kamil Matejuk

## Linki
[Dataset >>](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)  
[Raport (PDF) >>](./raport.pdf)  

## Uruchomienie
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
dvc init
dvc repro
```

Notebooki zostaną wykonane i zapisane w `results/` jako pliki HTML. Może to chwilę potrwać.

Uwaga: przed uruchomieniem `dvc repro` należy wykonać notebook `load_data.ipynb` (dane zostaną wówczas pobrane)
