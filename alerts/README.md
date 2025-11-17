# 🚨 Alerts Tutorials

Complete tutorials for receiving and processing astronomical alerts (gravitational waves, neutrinos, transients) via Kafka streams.

## ⚡ Quick Start

### 1. Setup Credentials (2 minutes)
```bash
# Edit credentials.txt and fill in your API keys
nano alerts/credentials.txt
```

### 2. Use in Any Notebook
```python
from credentials import load_credentials, get_credential

load_credentials()
slack_webhook = get_credential('SLACK_WEBHOOK')
```

## 📚 Tutorials

| Notebook | Service | Topic |
|----------|---------|-------|
| `skymaps.ipynb` | HEALPix | Sky localization maps analysis |
| `kafka.ipynb` | GCN | Gravitational wave alerts |
| `fink.ipynb` | Fink | ZTF transient alerts |
| `icecube.ipynb` | IceCube | Neutrino alerts |
| `igwn.ipynb` | IGWN | Gravitational wave network |
| `slack.ipynb` | Slack | Send alerts to Slack |
| `telegram.ipynb` | Telegram | Send alerts to Telegram |
| `mail.ipynb` | Email | Send alerts via email |
| `cook.ipynb` | Multi | Combine multiple services |

## 🔐 Credentials File (`credentials.txt`)

The file contains placeholders for all available services. Fill only what you need:

```
GCN_CLIENT_ID=your_id_here
GCN_CLIENT_SECRET=your_secret_here
SLACK_WEBHOOK=https://hooks.slack.com/...
# etc.
```

**Security:** This file is in `.gitignore` - your credentials are never committed! ✅

## 📖 Getting Credentials

- **GCN:** https://www.gcn.nasa.gov/kafka
- **Fink:** https://forms.gle/2td4jysT4e9pkf889
- **Slack:** https://api.slack.com/apps
- **Telegram:** https://t.me/botfather
- **GraceDB:** https://gracedb.ligo.org/

## 🎨 Plots

All notebooks use `plot_config.py` for high-quality, theme-compatible plots:
```python
from plot_config import setup_matplotlib_style
setup_matplotlib_style()  # Already auto-configured
```
