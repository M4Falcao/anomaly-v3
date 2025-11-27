
try:
    with open('debug_output.txt', 'r', encoding='utf-16') as f:
        lines = f.readlines()
except UnicodeError:
    with open('debug_output.txt', 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

debug_lines = [l.strip() for l in lines if 'DEBUG' in l]

with open('debug_summary.txt', 'w') as f:
    f.write('\n'.join(debug_lines))

print("Extracted lines:")
print('\n'.join(debug_lines))
