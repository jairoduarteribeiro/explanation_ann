#!/bin/bash

if [ ! -d ".venv" ]; then
  echo "Installing virtual environment..."
  python3 -m venv .venv

  echo "Activating virtual environment..."
  source .venv/bin/activate

  echo "Upgrading pip..."
  pip install -U pip

  echo "Installing dependencies..."
  pip install -r requirements.txt

  echo "Exiting virtual environment..."
  deactivate
fi