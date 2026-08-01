#!/usr/bin/env python3
"""Per-service RORT_* read closure: entrypoint -> transitive import closure -> flags read.

Board #165 Phase 0 / #265. Source-1 half of the flag-inventory intersection: for each
Railway service's Dockerfile CMD entrypoint, walk every Import/ImportFrom node (including
function-level lazy imports) to build the module closure under src/, then collect the
RORT_* tokens each closure module actually reads via os.getenv/os.environ.

READ-ONLY. Import reachability is an UPPER BOUND on reads, not proof of execution — see
docs/_active/Flag_Inventory_Read_vs_Set_165.md section 1. "Does not read it" is the strong
direction; "reads it" must be hand-verified before it is called a gap. Worker's container
HEALTHCHECK (Dockerfile.worker: `python src/engine_health.py`) is a SECOND entrypoint in
the same environment and is included below for that reason.

Usage:  python tools/flag_read_closure.py out.json    (run from the RoR_Trader repo root)
"""
import ast, os, re, json, sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

# module name (dotted, relative to src/) -> path
MODMAP = {}
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in ('__pycache__', '.git', 'node_modules')]
    for fn in filenames:
        if not fn.endswith('.py'):
            continue
        full = os.path.join(dirpath, fn)
        rel = os.path.relpath(full, ROOT)[:-3].replace(os.sep, '.')
        if rel.endswith('.__init__'):
            rel = rel[:-len('.__init__')]
        MODMAP[rel] = full
        # also allow "src.<mod>" form
        MODMAP['src.' + rel] = full

# --- flag reads per module -------------------------------------------------
GETENV = re.compile(
    r"""(?:os\.environ\.get|os\.getenv|_os\.getenv|_os_bt\.getenv|_os\.environ\.get|
         os\.environ\[|environ\.get|getenv)\s*\(?\s*['"](RORT_[A-Z0-9_]+)['"]""",
    re.X)
ANYTOK = re.compile(r"RORT_[A-Z0-9_]+")

def flags_in(path):
    try:
        src = open(path, encoding='utf-8', errors='replace').read()
    except OSError:
        return set(), set()
    # collapse line-continuations/newlines inside call parens for multiline getenv
    flat = re.sub(r"\(\s*\n\s*", "(", src)
    flat = re.sub(r"\n\s*", " ", flat)
    read = set(GETENV.findall(flat))
    mentioned = set(ANYTOK.findall(src))
    return read, mentioned

MOD_READ = {}
MOD_MENTION = {}
for mod, path in MODMAP.items():
    if mod.startswith('src.'):
        continue
    r, m = flags_in(path)
    MOD_READ[mod] = r
    MOD_MENTION[mod] = m

# --- import graph ----------------------------------------------------------
def imports_of(mod, path):
    out = set()
    try:
        tree = ast.parse(open(path, encoding='utf-8', errors='replace').read(), path)
    except SyntaxError:
        return out
    pkg = mod.rsplit('.', 1)[0] if '.' in mod else ''
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                base = pkg
                for _ in range(node.level - 1):
                    base = base.rsplit('.', 1)[0] if '.' in base else ''
                target = (base + '.' + node.module) if node.module else base
            else:
                target = node.module or ''
            if target:
                out.add(target)
                for a in node.names:
                    out.add(target + '.' + a.name)
            # also allow same-package sibling for `from . import x`
            if node.level and not node.module:
                for a in node.names:
                    out.add((pkg + '.' + a.name) if pkg else a.name)
    return out

def resolve(name):
    """map an import target onto a module in MODMAP, honouring src. prefix"""
    cands = [name]
    if name.startswith('src.'):
        cands.append(name[4:])
    else:
        cands.append('src.' + name)
    for c in cands:
        if c in MODMAP:
            p = MODMAP[c]
            rel = os.path.relpath(p, ROOT)[:-3].replace(os.sep, '.')
            if rel.endswith('.__init__'):
                rel = rel[:-len('.__init__')]
            return rel
    return None

def closure(entry):
    seen, stack = set(), [entry]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        p = MODMAP.get(m)
        if not p:
            continue
        for imp in imports_of(m, p):
            r = resolve(imp)
            if r and r not in seen:
                stack.append(r)
    return seen

# Railway service -> Dockerfile CMD entrypoint. Worker carries a second entrypoint
# (engine_health.py, its container HEALTHCHECK) which reads RORT_HEALTHCHECK_*.
EXTRA_ENTRIES = {'Worker': ['engine_health']}

SERVICES = {
    'Worker': 'worker',
    'Data Worker': 'data_worker',
    'batch-worker': 'batch_worker',
    'shadow-worker': 'shadow_worker',
    'api': 'api.main',
}

result = {}
for svc, entry in SERVICES.items():
    c = closure(entry)
    for extra in EXTRA_ENTRIES.get(svc, ()):
        c |= closure(extra)
    flags = set()
    prov = defaultdict(list)
    for m in sorted(c):
        for f in MOD_READ.get(m, ()):  # only real getenv reads
            flags.add(f)
            prov[f].append(m)
    result[svc] = {'closure_size': len(c), 'flags': sorted(flags),
                   'prov': {k: sorted(v) for k, v in prov.items()},
                   'modules': sorted(c)}

json.dump(result, open(sys.argv[1], 'w'), indent=1)
for svc, d in result.items():
    print(f"{svc:14s} modules={d['closure_size']:4d}  RORT_ reads={len(d['flags'])}")
