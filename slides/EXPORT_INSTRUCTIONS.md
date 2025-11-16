# Come Esportare Presentazione Keynote ad Alta Risoluzione

## Metodo 1: Esporta HTML con Alta Qualità (Consigliato)

1. **Apri la presentazione in Keynote**
2. **File → Export To → HTML**
3. **Nelle opzioni di esportazione:**
   - ✅ **Navigation Controls**: Include
   - ✅ **Playable on**: Devices without Keynote
   - ⚠️ **Image Quality**: Seleziona "Best" o "High" (non "Good" o "Medium")
   - ⚠️ **Resolution**: Se disponibile, seleziona 1920x1080 o superiore

4. **Salva nella cartella `slides/GCN_acme/`**
   - Sostituisci i file esistenti

## Metodo 2: Esporta come PDF ad Alta Risoluzione (Alternativa)

Se Keynote non permette di aumentare la risoluzione HTML, esporta come PDF:

1. **File → Export To → PDF**
2. **Image Quality**: Best (300 DPI o superiore)
3. Salva come `slides/GCN_acme_presentation.pdf`

Poi modifica `presentation.md` per usare PDF viewer invece di HTML:

```html
<iframe src="slides/GCN_acme_presentation.pdf" 
        width="100%" 
        height="800px" 
        style="border: 1px solid #e5e7eb;">
</iframe>
```

## Metodo 3: Usa Immagini ad Alta Risoluzione

Esporta ogni slide come immagine PNG ad alta risoluzione:

1. **File → Export To → Images**
2. **Format**: PNG
3. **Resolution**: Scegli dimensione custom (es. 2560x1440 o 3840x2160 per 4K)
4. Salva in `slides/GCN_acme/images/`

Poi crea uno slideshow con reveal.js o simili.

## Metodo 4: Video Export (Massima Qualità)

Per la massima qualità visiva:

1. **File → Export To → Movie**
2. **Resolution**: 4K (3840x2160) o almeno 1080p
3. **Codec**: H.264 (alta qualità)
4. Carica su YouTube o Vimeo
5. Embed nel sito

```html
<iframe width="1280" height="720" 
        src="https://www.youtube.com/embed/YOUR_VIDEO_ID" 
        frameborder="0" allowfullscreen>
</iframe>
```

## Verifica Risoluzione Attuale

Per controllare la risoluzione delle immagini esportate:
```bash
cd slides/GCN_acme/assets/
file *.jpg *.png | head -5
```

Se vedi dimensioni come "1024x768" o "800x600", devi ri-esportare con qualità maggiore.

## Nota

Le presentazioni Keynote HTML esportate hanno una **risoluzione fissa** determinata al momento dell'export. Non è possibile aumentarla modificando solo il file HTML - devi ri-esportare da Keynote.
