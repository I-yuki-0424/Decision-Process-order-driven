import os
import sys
import re

def check_file(filepath):
    errors = []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.splitlines()
    except Exception:
        return errors

    # Check 1: "VERIFIED_SUCCESS" literal
    # We construct the string dynamically so this script doesn't fail its own check
    banned = "VERIFIED" + "_" + "SUCCESS"
    if f'"{banned}"' in content or f"'{banned}'" in content:
        errors.append(f"{filepath}: Contains banned literal {banned}")

    # Check 2: script names containing retrieve, verify, or real
    filename = os.path.basename(filepath)
    if any(w in filename.lower() for w in ['retrieve', 'verify', 'real']) and filepath.endswith('.py'):
        # Needs I/O call, environment step call, or model forward pass
        has_io = re.search(r'\b(open|request|jnp\.load|pickle\.load|save_model_checkpoint|load_model_checkpoint)\(', content)
        has_env = re.search(r'\b(env\.step|raw_env\.step)\(', content)
        has_forward = re.search(r'\b(forward|__call__|forward_hierarchical_transformer)\(', content)

        if not (has_io or has_env or has_forward):
            errors.append(f"{filepath}: Name contains retrieve/verify/real but has no I/O, env step, or model forward pass")

    # Check 3: Dict literals for metrics
    in_dict = False
    dict_name = ""
    num_literals = 0
    for i, line in enumerate(lines):
        # Extremely simplified check for dict assignment
        match = re.match(r'^\s*([a-zA-Z0-9_]*(metrics|results|rollout|benchmark)[a-zA-Z0-9_]*)\s*=\s*\{', line)
        if match:
            in_dict = True
            dict_name = match.group(1)
            num_literals = 0
            continue

        if in_dict:
            if '}' in line:
                in_dict = False
                if num_literals > 2:
                    errors.append(f"{filepath}:{i+1}: Dict {dict_name} contains >2 numeric literals without # SOURCE")
            else:
                # Count numeric literals if no # SOURCE:
                if '# SOURCE:' not in line:
                    if re.search(r':\s*[-+]?[0-9]*\.?[0-9]+', line):
                        num_literals += 1

    # Check 4: metric-like keyword argument set to a bare float literal, e.g. `noise_recovery_rate=0.85,`
    # (a measured value must be computed; a constant with a metric name is a fabricated result)
    kw = re.compile(r'^\s+([A-Za-z0-9_]*(rate|resilience|accuracy|score|reward|success)[A-Za-z0-9_]*)\s*=\s*[-+]?[0-9]+\.[0-9]+\s*(,|(?=\s*$))\s*(#.*)?$')
    for i, line in enumerate(lines):
        m = kw.match(line)
        # keyword-argument context only: the line ends with a comma, or the next line closes the call
        is_kwarg = bool(m) and (line.split('#')[0].rstrip().endswith(',') or
                                (i + 1 < len(lines) and lines[i + 1].lstrip().startswith(')')))
        if is_kwarg and '# SOURCE:' not in line:
            errors.append(f"{filepath}:{i+1}: metric-like keyword {m.group(1)} set to a numeric literal (compute it, or annotate '# SOURCE:')")

    return errors


def check_json(filepath):
    """Result files must not carry the banned status literal (the .py check cannot see committed outputs)."""
    banned = "VERIFIED" + "_" + "SUCCESS"
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            head = f.read(2000000)
    except Exception:
        return []
    return [f"{filepath}: JSON output contains banned literal {banned}"] if banned in head else []


all_errors = []
for root, dirs, files in os.walk('.'):
    if '.git' in root or '__pycache__' in root or 'venv' in root:
        continue
    norm = root.replace(os.sep, '/')
    if 'output_remote' in norm or 'quarantine_fabricated' in norm:
        continue
    for file in files:
        if file.endswith('.json') and (norm.startswith('./output') or norm.startswith('./kaggle')):
            all_errors.extend(check_json(os.path.join(root, file)))
        if file.endswith('.py') or file.endswith('.sh'):
            if file == 'pre_commit_ban_check.py' or file == 'check_banned_patterns.py':
                continue
            filepath = os.path.join(root, file)
            all_errors.extend(check_file(filepath))

if all_errors:
    print("Pre-commit check failed:")
    for e in all_errors:
        print(e)
    sys.exit(1)
print("Pre-commit check passed")
