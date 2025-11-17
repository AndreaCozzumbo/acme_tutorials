"""
🔐 Credentials Configuration Loader

This module reads credentials from 'credentials.txt' file.
The credentials file uses KEY=VALUE format for easy manual editing.

⚠️ IMPORTANT: 
- DO NOT commit credentials.txt to version control
- Add 'credentials.txt' to your .gitignore

Usage in notebooks:
    from credentials import get_credential, load_credentials
    
    # Load credentials at the start
    load_credentials()
    
    # Access individual credentials
    slack_webhook = get_credential('SLACK_WEBHOOK')
    gcn_id = get_credential('GCN_CLIENT_ID')
"""

import os
from pathlib import Path


# Store loaded credentials in memory
_CREDENTIALS = {}
_CREDENTIALS_LOADED = False


def load_credentials(filepath='credentials.txt'):
    """
    Load credentials from a text file.
    
    Parameters:
    -----------
    filepath : str
        Path to credentials file (default: 'credentials.txt' in same directory)
    
    Raises:
    -------
    FileNotFoundError if credentials file doesn't exist
    """
    global _CREDENTIALS, _CREDENTIALS_LOADED
    
    # If not provided, look for credentials.txt in the same directory as this file
    if filepath == 'credentials.txt':
        filepath = Path(__file__).parent / 'credentials.txt'
    else:
        filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Credentials file not found: {filepath}\n"
            f"Please create '{filepath}' and fill in your credentials."
        )
    
    print(f"📖 Loading credentials from: {filepath}")
    
    # Parse the credentials file
    _CREDENTIALS.clear()
    with open(filepath, 'r') as f:
        for line in f:
            # Remove whitespace and skip empty lines
            line = line.strip()
            
            # Skip comments
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE format
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                _CREDENTIALS[key] = value
    
    _CREDENTIALS_LOADED = True
    print(f"✅ Loaded {len(_CREDENTIALS)} credentials")
    return _CREDENTIALS


def get_credential(key, default=None):
    """
    Retrieve a single credential value.
    
    Parameters:
    -----------
    key : str
        Credential key (e.g., 'SLACK_WEBHOOK', 'GCN_CLIENT_ID')
    default : str, optional
        Default value if credential is not found or empty
    
    Returns:
    --------
    str : The credential value
    
    Raises:
    -------
    ValueError if credential is not configured and no default provided
    """
    if not _CREDENTIALS_LOADED:
        raise RuntimeError(
            "Credentials not loaded! Call load_credentials() first:\n"
            "  from credentials import load_credentials\n"
            "  load_credentials()"
        )
    
    value = _CREDENTIALS.get(key, default)
    
    if not value:
        raise ValueError(
            f"❌ Credential '{key}' is not configured.\n"
            f"Please fill in '{key}' in credentials.txt"
        )
    
    return value


def check_credential(key):
    """
    Check if a credential is configured (non-empty).
    
    Parameters:
    -----------
    key : str
        Credential key
    
    Returns:
    --------
    bool : True if credential exists and is non-empty
    """
    if not _CREDENTIALS_LOADED:
        return False
    
    return bool(_CREDENTIALS.get(key, '').strip())


def list_credentials():
    """
    List all loaded credentials (without showing actual values for security).
    
    Returns:
    --------
    dict : Dictionary with credential keys and whether they're configured
    """
    if not _CREDENTIALS_LOADED:
        return {"status": "Credentials not loaded"}
    
    status = {}
    for key, value in _CREDENTIALS.items():
        status[key] = "✅ Configured" if value.strip() else "⚠️ Empty"
    
    return status
