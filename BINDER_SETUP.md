# Binder Configuration for ACME Tutorials

## Problemi Risolti

### 1. Pacchetti Python Mancanti
**Problema**: I notebook mostravano errori "module not found" quando eseguiti nel browser tramite Binder o Thebe.

**Soluzione**: Aggiunti tutti i pacchetti necessari in:
- `binder/requirements.txt` - ora include `GWFish` e `lalsuite`
- `binder/environment.yml` - configurazione conda aggiornata

### 2. Live Code Button Non Funzionante
**Problema**: Il pulsante "Live Code" (Thebe) non si connetteva correttamente a Binder.

**Soluzione**: 
- Aggiunta configurazione esplicita di Binder in `_config.yml`
- Creato file `.thebe_config` con le impostazioni corrette
- Aggiornati i riferimenti al repository

## Pacchetti Installati

### Pacchetti Scientifici Core
- numpy==1.25
- scipy, matplotlib, pandas
- astropy, astropy-healpix, healpy
- ligo.skymap

### Pacchetti per Tutorial
- **GWFish** (installato da GitHub)
- **lalsuite** (per analisi onde gravitazionali)
- corner, tqdm, tables, sympy
- xmltodict, h5py, lxml

### Jupyter Environment
- jupyterlab, ipykernel, ipywidgets
- notebook<7.0 (compatibilità Binder)

## Come Testare

### Test Locale
1. Build del Jupyter Book:
   ```bash
   jupyter-book build .
   ```

2. Apri `_build/html/index.html` nel browser

3. Clicca su "Live Code" in un notebook e attendi la connessione a Binder

### Test su Binder Diretto
Dopo aver fatto push su GitHub, testa il badge Binder:

```markdown
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/samueleronchini/acme_tutorials/main)
```

### Verifica Pacchetti nel Browser
Una volta connesso a Thebe/Binder:

```python
# Test imports
import numpy as np
import GWFish
import lal
import healpy as hp
print("✅ Tutti i pacchetti installati correttamente!")
```

## Limitazioni Note

### Kafka Streaming
I pacchetti `confluent-kafka` e `gcn-kafka` NON sono installati su Binder perché:
- Richiedono librerie di sistema complesse
- Hanno restrizioni di rete su Binder
- Sono troppo pesanti per l'ambiente Binder

**Workaround**: Per i tutorial Kafka, gli utenti devono:
1. Usare installazione locale
2. Oppure vedere gli output pre-calcolati nei notebook

## File di Configurazione Modificati

1. **binder/requirements.txt** - Aggiunti GWFish e lalsuite
2. **binder/environment.yml** - Aggiornate dipendenze conda
3. **binder/postBuild** - Aggiunta verifica installazione
4. **_config.yml** - Configurazione Thebe/Binder migliorata
5. **.thebe_config** - Nuovo file per configurazione Thebe

## Tempi di Build su Binder

**Prima build**: ~15-20 minuti (Binder deve costruire l'ambiente)
**Build successive**: ~2-5 minuti (usa cache)

## Troubleshooting

### Il pulsante "Live Code" non appare
- Verifica che il file sia un `.ipynb` o una pagina MyST con code cells
- Controlla che `thebe: true` sia in `_config.yml`

### Binder fallisce al build
- Controlla i log su mybinder.org
- Verifica che tutti i pacchetti in `requirements.txt` siano validi
- Alcuni pacchetti potrebbero richiedere versioni specifiche di Python

### I pacchetti sembrano installati ma danno errore
- Aspetta che Binder completi il build (barra di progresso al 100%)
- Riavvia il kernel (icona circolare in alto)
- Verifica la versione di Python (deve essere 3.11)

## Prossimi Passi

1. **Commit e push** delle modifiche:
   ```bash
   git add binder/ _config.yml .thebe_config
   git commit -m "Fix Binder and Thebe configuration for live code"
   git push
   ```

2. **Test completo**: Attendi il build su mybinder.org (prima volta)

3. **Aggiorna documentazione**: Aggiungi badge Binder al README principale

## Risorse

- [Jupyter Book Thebe Documentation](https://jupyterbook.org/en/stable/interactive/thebe.html)
- [Binder Documentation](https://mybinder.readthedocs.io/)
- [GWFish Repository](https://github.com/janosch314/GWFish)
