# Introduction

Welcome to the **Multi-Messenger Alert System** tutorials! This comprehensive series guides you through receiving, processing, and responding to astronomical alerts from multiple sources in real-time.

## What You'll Learn

This tutorial suite covers:

- **Gravitational Wave Detection**: Process alerts from LIGO, Virgo, and KAGRA via GCN and IGWN
- **Neutrino Observations**: Real-time neutrino data from IceCube
- **Transient Astronomy**: Rapidly changing objects detected by Fink and other surveys
- **Kafka Streaming**: Real-time data streaming and message brokering
- **Sky Localization**: Analyze and visualize sky maps (HEALPix format)
- **Multi-Channel Alerts**: Send notifications to Slack, Telegram, and Email
- **Data Integration**: Combine data from multiple sources into unified pipelines

## Quick Start 

### Step 1: Configure Credentials
```bash
# Copy the template and fill in your API keys
cp alerts/credentials.example.txt alerts/credentials.txt
nano alerts/credentials.txt
```

### Step 2: Use in Notebooks
```python
from credentials import load_credentials, get_credential

# Load credentials from file
load_credentials()

# Get a specific credential
webhook = get_credential('SLACK_WEBHOOK')
bot_token = get_credential('TELEGRAM_BOT_TOKEN')
```

### Step 3: Choose Your Tutorial
Navigate to any notebook below to get started!

## How to Run the Notebooks

### Option 1: Local Jupyter (Recommended for Development)

Run notebooks locally with full access to your machine:

```bash
# Clone the repository
git clone https://github.com/samueleronchini/acme_tutorials.git
cd acme_tutorials

# Create and activate virtual environment (recommended)
python -m venv acme_tutorials
source acme_tutorials/bin/activate  # On Windows: venv\Scripts\activate

# Install Jupyter if not already installed
pip install jupyter jupyterlab

# Install dependencies
pip install -r binder/requirements.txt

# Start Jupyter
jupyter notebook

# Or use JupyterLab for a better experience
jupyter lab
```

Then open any `.ipynb` notebook and run cells with **Shift+Enter**.

### Option 2: Binder (Zero Installation Required)

Run notebooks directly in your browser without installing anything:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/samueleronchini/acme_tutorials/main)

**Steps:**
1. Click the Binder badge above (or in the README)
2. Wait for the environment to build (~2 minutes first time)
3. Navigate to the `alerts/` folder
4. Click any `.ipynb` notebook to open it
5. Run cells with **Shift+Enter**

**Pros:** No setup needed, runs in browser  
**Cons:** Limited to tutorial execution only (no data persistence between sessions)

### Option 3: Google Colab (For Quick Exploration)

Upload and run notebooks in Google Colab:

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Click **File → Open notebook**
3. Paste GitHub URL: `https://github.com/samueleronchini/acme_tutorials`
4. Select a notebook from `alerts/` folder
5. Run cells with **Shift+Enter**

**Installation in Colab:**
```python
# In the first cell, install dependencies
!pip install slack-sdk hop-client healpy astropy
```

**Pros:** Free GPU available, easy to share  
**Cons:** Need to reconfigure credentials each session, files not saved

### Cell Execution Tips

**Run a single cell:**
- Click the cell to select it
- Press **Shift+Enter** (or click the ▶ button)

**Run all cells:**
- Use **Kernel → Restart & Run All** menu

**Stop execution:**
- Press **Esc** to exit edit mode, then press **I** twice quickly
- Or use **Kernel → Interrupt**

**Clear outputs:**
- Use **Cell → Clear Output** to remove cell results
- **Cell → Clear All Output** for entire notebook

## Tutorial Structure

### Core Alert Systems
- **[kafka.ipynb](alerts/kafka)** - GCN Kafka broker & gravitational wave alerts
- **[igwn.ipynb](alerts/igwn)** - IGWN network integration and event filtering
- **[scimma.ipynb](alerts/scimma)** - Hopskotch/SCIMMA infrastructure for multi-messenger data

### Alert Sources
- **[icecube.ipynb](alerts/icecube)** - IceCube neutrino real-time alerts
- **[fink.ipynb](alerts/fink)** - Fink broker for transient follow-ups

### Sky Analysis
- **[skymaps.ipynb](alerts/skymaps)** - HEALPix sky localization maps, visualization & analysis

### Notification Channels
- **[slack.ipynb](alerts/slack)** - Send structured alerts to Slack channels
- **[telegram.ipynb](alerts/telegram)** - Push notifications to Telegram bots
- **[mail.ipynb](alerts/mail)** - Email alerts with attachments

### Advanced Integration
- **[cook.ipynb](alerts/cook)** - Combine multiple services into unified pipelines

## Security & Credentials

All authentication is handled through the centralized **`credentials.txt`** file:

- **Safe**: Automatically protected by `.gitignore` - your keys never get committed
- **Simple**: Just `KEY=VALUE` format - no complex configuration
- **Flexible**: Load what you need, leave the rest blank
- **Template**: Start with `credentials.example.txt` to see all available services

```python
# Check what's configured
from credentials import list_credentials
list_credentials()

# Get a credential with fallback
token = get_credential('SLACK_WEBHOOK', default=None)
if token is None:
    print("Please configure SLACK_WEBHOOK in credentials.txt")
```

## Supported Services

| Service | Protocol | Purpose | Tutorial |
|---------|----------|---------|----------|
| **GCN** | Kafka | Gamma-ray, gravitational wave, neutrino alerts | [kafka.ipynb](alerts/kafka) |
| **IGWN** | Kafka | Gravitational wave network events | [igwn.ipynb](alerts/igwn) |
| **Hopskotch** | Kafka | Multi-messenger Kafka infrastructure | [scimma.ipynb](alerts/scimma) |
| **IceCube** | REST API | High-energy neutrino events | [icecube.ipynb](alerts/icecube) |
| **Fink** | Kafka | Transient alert broker | [fink.ipynb](alerts/fink) |
| **Slack** | Webhook | Real-time channel notifications | [slack.ipynb](alerts/slack) |
| **Telegram** | Bot API | Direct messaging & groups | [telegram.ipynb](alerts/telegram) |
| **Email** | SMTP | Email notifications | [mail.ipynb](alerts/mail) |


## License

This tutorial suite is part of the ACME project. Author: [Samuele Ronchini](https://samueleronchini.github.io)

---

**Ready to start?** Pick any tutorial above and dive in! 🚀
