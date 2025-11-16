# 🔐 Managing Secrets and Credentials

## ⚠️ IMPORTANT: Never commit secrets to Git!

This repository uses external services that require authentication:
- **GCN Kafka** (for receiving gravitational wave alerts)
- **Slack** (for posting notifications)
- **Telegram** (for posting notifications)

## Setup Instructions

### 1. Create your `.env` file

```bash
# Copy the example file
cp .env.example .env

# Edit with your credentials
nano .env  # or use your favorite editor
```

### 2. Get your credentials

#### GCN Kafka Credentials
1. Visit https://gcn.nasa.gov/quickstart
2. Create an account or sign in
3. Generate your client ID and secret
4. Add them to `.env`:
   ```
   GCN_CLIENT_ID=your_client_id_here
   GCN_CLIENT_SECRET=your_client_secret_here
   ```

#### Slack Bot Token
1. Visit https://api.slack.com/apps
2. Create a new app or select existing
3. Go to "OAuth & Permissions"
4. Copy the Bot User OAuth Token (starts with `xoxb-`)
5. Add to `.env`:
   ```
   SLACK_BOT_TOKEN=xoxb-your-token-here
   SLACK_CHANNEL_ID=your-channel-id-here
   ```

#### Telegram Bot Token
1. Open Telegram and search for @BotFather
2. Send `/newbot` and follow instructions
3. Copy the token provided
4. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your-telegram-token-here
   ```

### 3. Load credentials in notebooks

The notebooks are configured to read from environment variables:

```python
import os

# Credentials are loaded automatically
GCN_CLIENT_ID = os.getenv('GCN_CLIENT_ID', 'YOUR_CLIENT_ID_HERE')
GCN_CLIENT_SECRET = os.getenv('GCN_CLIENT_SECRET', 'YOUR_CLIENT_SECRET_HERE')
```

### 4. Alternative: Hardcode (NOT recommended for Git repos)

If running locally and NOT committing to Git, you can hardcode:

```python
# Only for local use - DO NOT commit this!
GCN_CLIENT_ID = 'your_actual_id'
GCN_CLIENT_SECRET = 'your_actual_secret'
```

## Security Notes

- ✅ `.env` is in `.gitignore` - it won't be committed
- ✅ `.env.example` is a safe template with no real secrets
- ❌ **NEVER** commit real tokens/secrets to Git
- ❌ **NEVER** share your `.env` file publicly

## If you accidentally committed secrets

If you already committed secrets to Git:

1. **Revoke the compromised credentials immediately**
   - GCN: Generate new credentials at https://gcn.nasa.gov
   - Slack: Regenerate token in app settings
   - Telegram: Create a new bot with @BotFather

2. **Remove from Git history** (advanced):
   ```bash
   # Use git filter-branch or BFG Repo-Cleaner
   # See: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository
   ```

3. **Update your `.env` with new credentials**

## GitHub Push Protection

GitHub automatically scans for leaked secrets. If push is rejected:

```
remote: push declined due to repository rule violations
```

This means GitHub detected a secret in your commit. Follow the steps above to fix it.
