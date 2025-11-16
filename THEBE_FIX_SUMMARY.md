# 🔧 Thebe Live Code - Riepilogo Fix Applicati

## 📋 Problema Originale
- **Sintomo 1:** Pulsante "Live Code" rimane bloccato su "starting..."
- **Sintomo 2:** Pulsanti "run" non compaiono dopo aver cliccato "Live Code"
- **Sintomo 3:** kafka_1.ipynb funziona, ma tutti gli skymaps_*.ipynb si bloccano

## 🎯 Causa Root
**Il kernelspec nei notebook skymaps era impostato su "bat" invece di "python3"**

Quando hai eseguito i notebook localmente con un environment Python chiamato "bat", Jupyter ha salvato questo nome nel metadata del notebook. Quando Thebe prova a connettersi a Binder, cerca un kernel chiamato "bat" che non esiste nell'ambiente Binder, causando il blocco.

### Evidenza
```bash
# Prima del fix:
alerts/skymaps_1_intro.ipynb: name=bat
alerts/skymaps_2_operations.ipynb: name=bat
alerts/skymaps_3_multiorder.ipynb: name=bat
alerts/skymaps_4_advanced.ipynb: name=bat

# kafka_1 funzionava perché:
alerts/kafka_1.ipynb: name=python3 ✅
```

## ✅ Soluzioni Applicate

### 1. Fix Kernelspec (SOLUZIONE PRINCIPALE)
**File modificati:** Tutti i notebook `alerts/skymaps*.ipynb`

**Cambiamento:**
```json
// PRIMA
"kernelspec": {
  "name": "bat",
  "display_name": "bat"
}

// DOPO
"kernelspec": {
  "name": "python3",
  "display_name": "Python 3 (ipykernel)",
  "language": "python"
}
```

**Comando usato:**
```python
python3 << 'EOF'
import json, glob
for nb_path in glob.glob('alerts/skymaps*.ipynb'):
    with open(nb_path, 'r') as f:
        nb = json.load(f)
    nb['metadata']['kernelspec'] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    }
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
EOF
```

### 2. Rimozione Template Problematico
**File rimossi:** `_templates/layout.html`

**Motivo:** Causava errore su GitHub Actions:
```
jinja2.exceptions.TemplateNotFound: toggle-primary-sidebar.html
```

Il template personalizzato era in conflitto con `pydata_sphinx_theme`.

### 3. Configurazione Binder (già corretta in precedenza)
**File:** `binder/environment.yml`

Contiene tutti i pacchetti necessari:
- astropy>=5.0
- healpy>=1.16
- numpy=1.25
- matplotlib
- gcn-kafka
- ligo.skymap
- GWFish (da GitHub)

## 📊 Commit Applicati

1. **"Fix Binder config, add gcn-kafka, split skymaps into 4 parts"**
   - Aggiunti pacchetti mancanti in binder/environment.yml
   - Divisi notebook skymaps in 4 parti più leggere
   - Aggiunto gcn-kafka

2. **"Fix Thebe configuration with custom binderOptions script"**
   - Tentativo con script JavaScript (poi rimosso)

3. **"Fix kernelspec from bat to python3 in all skymaps notebooks"**
   - Fix principale del kernelspec

4. **"Remove _templates to fix GitHub Actions build error"**
   - Rimozione template problematico

## 🧪 Come Testare

### Locale (con http.server)
```bash
cd _build/html
python3 -m http.server 8001
# Apri http://localhost:8001/alerts/skymaps_1_intro.html
```

### Produzione (GitHub Pages)
1. Vai su: https://samueleronchini.github.io/acme_tutorials/alerts/skymaps_1_intro.html
2. Clicca il pulsante "Live Code" 🚀 (in alto a destra)
3. Aspetta che mostri "Ready" (15-20 min la prima volta, poi 2-5 min)
4. I pulsanti "run" appariranno sotto ogni cella di codice
5. Clicca "run" su una cella per eseguirla

## ⚠️ Note Importanti

### Perché Live Code non funziona con file://
Thebe richiede HTTPS/HTTP per motivi di sicurezza del browser. Da `file://` locale non può connettersi a Binder.

### Prima Build di Binder
La prima volta che un utente clicca "Live Code":
- Binder deve costruire l'ambiente dalla repo GitHub
- Tempo richiesto: **15-20 minuti**
- Le volte successive (entro ~1 settimana): **2-5 minuti** (usa cache)

### Differenza tra kafka_1 e skymaps
- **kafka_1:** 4 celle di codice, import leggeri → "ready" veloce
- **skymaps_1:** 7 celle di codice, import pesanti (healpy, matplotlib, astropy) → più lento

### Come Evitare il Problema in Futuro
**MAI eseguire notebook con environment chiamati con nomi custom** prima di committarli!

**Opzione 1:** Usa sempre environment con kernel standard:
```bash
# Crea environment ma usa kernel python3
conda create -n myenv python=3.11
conda activate myenv
python -m ipykernel install --user --name python3 --display-name "Python 3"
```

**Opzione 2:** Resetta il kernelspec prima di committare:
```python
import json
with open('notebook.ipynb', 'r') as f:
    nb = json.load(f)
nb['metadata']['kernelspec'] = {
    "name": "python3",
    "display_name": "Python 3 (ipykernel)",
    "language": "python"
}
with open('notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
```

## 🔍 Debug Console JavaScript

Per vedere cosa sta facendo Thebe:
1. Apri DevTools del browser (F12 o Cmd+Option+I)
2. Vai alla tab "Console"
3. Clicca "Live Code"
4. Vedrai log come:
   ```
   Thebe: Connecting to Binder...
   Binder: Building environment...
   Thebe: Kernel ready!
   ```

## 📚 Riferimenti

- Jupyter Book Thebe: https://jupyterbook.org/en/stable/interactive/launchbuttons.html#thebe
- Binder: https://mybinder.org
- GitHub Actions: https://github.com/samueleronchini/acme_tutorials/actions
- Deployed site: https://samueleronchini.github.io/acme_tutorials

---

**Fix completato il:** 16 novembre 2025  
**Status:** ✅ Build completato, pronto per il test su GitHub Pages
