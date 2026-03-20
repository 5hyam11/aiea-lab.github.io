#!/usr/bin/env bash
# setup_and_push.sh
# Run this once from inside your langchain_logic/ folder to:
#   1. Create a venv and install deps
#   2. Initialize a git repo
#   3. Make the first commit
#   4. Push to GitHub (you supply the remote URL)
#
# Usage:
#   chmod +x setup_and_push.sh
#   ./setup_and_push.sh https://github.com/YOUR_USERNAME/langchain-logic-lm.git

set -e

REMOTE_URL="${1}"

echo "=== 1. Creating virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== 2. Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== 3. Initialising git repo ==="
git init
echo "venv/" >> .gitignore
echo ".env"  >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore

git add .
git commit -m "Initial commit: Logic-LM with LangChain + pure-Python Prolog"

if [ -n "$REMOTE_URL" ]; then
  echo "=== 4. Pushing to GitHub ==="
  git remote add origin "$REMOTE_URL"
  git branch -M main
  git push -u origin main
  echo "Done! Repo pushed to $REMOTE_URL"
else
  echo "No remote URL provided — skipping push."
  echo "To push later:  git remote add origin <URL> && git push -u origin main"
fi
