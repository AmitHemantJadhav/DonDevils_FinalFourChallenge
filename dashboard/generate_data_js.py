#!/usr/bin/env python3
"""
generate_data_js.py - Embed CSV text into a JS file for the dashboard.
Runs in under 1 second — just reads raw text, no parsing.

Usage: python3 dashboard/generate_data_js.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw')
OUT = os.path.join(ROOT, 'dashboard', 'data.js')

train_text = open(os.path.join(RAW, 'NCAA_Seed_Training_Set2.0.csv')).read()
test_text = open(os.path.join(RAW, 'NCAA_Seed_Test_Set2.0.csv')).read()

with open(OUT, 'w') as f:
    f.write('// Auto-generated — run: python3 dashboard/generate_data_js.py\n')
    f.write('const TRAINING_CSV = `\n')
    f.write(train_text)
    f.write('`;\n\n')
    f.write('const TEST_CSV = `\n')
    f.write(test_text)
    f.write('`;\n')

print(f'Done! {os.path.getsize(OUT) / 1024:.0f} KB written to dashboard/data.js')
