# Cameron's AI Assistant — PythonAnywhere Deployment Guide

## Overview
A RAG-powered chatbot backed by your resume, research, and bio.
Built with Flask + TF-IDF retrieval + Claude API.

---

## Step 1: Upload Files to PythonAnywhere

1. Log into pythonanywhere.com
2. Go to **Files** tab
3. Create a folder: `/home/<yourusername>/cameron_ai/`
4. Upload all files maintaining this structure:

```
cameron_ai/
├── app.py
├── rag.py
├── requirements.txt
├── documents/
│   ├── resume.txt        ← REPLACE with your actual resume
│   ├── bio.txt           ← REPLACE with your actual bio
│   ├── research.txt      ← REPLACE with your actual research summaries
│   └── [add more .txt files as needed]
└── templates/
    └── index.html
```

---

## Step 2: Install Dependencies

Open a **Bash console** on PythonAnywhere and run:

```bash
cd ~/cameron_ai
pip3.10 install --user flask scikit-learn numpy requests
```

---

## Step 3: Set Up the Web App

1. Go to the **Web** tab in PythonAnywhere
2. Click **Add a new web app**
3. Choose **Manual configuration** (not Flask quickstart)
4. Choose **Python 3.10**

In the web app config:
- **Source code**: `/home/<yourusername>/cameron_ai`
- **Working directory**: `/home/<yourusername>/cameron_ai`
- **WSGI file**: Click the link and replace ALL content with:

```python
import sys
sys.path.insert(0, '/home/<yourusername>/cameron_ai')
from app import app as application
```

---

## Step 4: Set Environment Variables

In the **Web** tab, scroll to **Environment variables** and add:

```
ANTHROPIC_API_KEY = your_actual_api_key_here
FLASK_SECRET = any_random_string_here
```

---

## Step 5: Add Your Documents

Replace the placeholder .txt files in `/documents/` with your actual content:

- **resume.txt** — Paste your full resume as plain text
- **bio.txt** — Your professional bio
- **research.txt** — Summaries of your research projects and publications
- Add **coursework.txt**, **linkedin.txt**, or any other files — the system auto-loads all .txt files

The more content you add, the better the retrieval quality.

---

## Step 6: Reload and Test

1. In the **Web** tab, click the green **Reload** button
2. Visit `https://<yourusername>.pythonanywhere.com`
3. Test with: "What research has Cameron published?"
4. Check `/health` endpoint to confirm documents are loaded

---

## Updating Documents (No Restart Needed)

If you update a document file, POST to `/refresh` to reload without restarting:
```bash
curl -X POST https://<yourusername>.pythonanywhere.com/refresh
```

---

## Generating Your QR Code

Once deployed, go to https://qr-code-generator.com and paste your URL.
Put the QR code on your final presentation slide.

---

## Troubleshooting

**500 errors**: Check the error log in the Web tab
**No documents loaded**: Visit `/health` and check `documents_loaded`
**API errors**: Verify ANTHROPIC_API_KEY is set correctly in environment variables
**Slow responses**: Normal on free tier — upgrade to paid for faster response times

---

## Free Tier Limits (pythonanywhere.com)
- 1 web app
- Sleeps after inactivity (first request after sleep is slow — just reload)
- CPU usage limits apply but are fine for demo purposes
- Upgrade to Hacker plan ($5/mo) for always-on + custom domain
