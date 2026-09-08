"""Dump ``output2.txt`` to stdout, tolerating UTF-16 or UTF-8 encoding.

PowerShell redirection writes UTF-16 by default, so the file is tried with that
encoding first and falls back to UTF-8.

Usage:
    python scripts/tools/print_output_log.py
"""

try:
    with open('output2.txt', 'r', encoding='utf-16') as f:
        print(f.read())
except Exception as e:
    print(e)
    try:
        with open('output2.txt', 'r', encoding='utf-8') as f:
            print(f.read())
    except OSError:
        pass
