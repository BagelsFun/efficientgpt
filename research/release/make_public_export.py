"""Build a sanitised, publication-candidate export of the repository WITHOUT touching this repository's history.

    python -m research.release.make_public_export --out ../efficientgpt-public [--ref HEAD] [--validate]
        [--author-name "..." --author-email "...@users.noreply.github.com"] [--keep-internal] [--deny WORD ...] [--allow WORD ...]

What it does (nothing is pushed or published):
1. `git archive <ref>`: exactly the committed files of one commit (no history, no untracked or ignored files).
2. Drops owner-internal planning documents (INTERNAL below) unless --keep-internal; each is listed in the manifest.
3. Redacts home-directory paths in text files (UTF-8, or UTF-16 with a BOM; line endings and encoding are preserved):
   `<drive>:\\Users\\<name>`, `/c/Users/<name>`, `/Users/<name>`, `/home/<name>`, URL-encoded `C%3A%5CUsers%5C<name>` and
   `torchinductor_<name>` become `<HOME>` / `torchinductor_<user>`; the `origin` repository slug (`owner/name`) becomes
   `<private-repository>`; this fork's own commit / tag ids (full or abbreviated, >= 7 hex digits) become `<private-commit>`
   (upstream nanoGPT ids and 64-hex digests are left alone). Every replacement is counted per file.
4. Writes RELEASE_MANIFEST.json (exclusions, redaction counts, scan results) into the export.
5. `git init` + ONE commit authored by --author-name / --author-email (placeholders by default, to be replaced by the owner's
   chosen public identity before publishing: `git commit --amend --reset-author`). No tags, no remote, no signing.
6. Scans the export and fails closed: the e-mails of this fork's own commits and of `git config user.email`, the local
   user name, the home-folder name, the computer name, the `origin` owner and any --deny word (all case-insensitive, also
   searched as raw UTF-8 / UTF-16LE bytes in every file), home paths, credentials, model/data files and any text file that
   cannot be decoded.
7. --validate: clones the export to a temporary directory, copies in the hash-checked test inputs (as
   research.release.clean_checkout does), runs all test suites there, deletes the clone. Only counts and return codes go
   into the manifest (test logs contain local paths). Exit status is non-zero if the scan or any suite fails.
"""

import argparse
import codecs
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

INTERNAL = [  # written for the owner (planning, career strategy); not part of a technical release by default
    'MISSION_STATUS.md', 'NEXT_STEPS.md', 'RESEARCH_BACKLOG.md', 'CAREER_VALUE_REPORT.md',
    'reports/CAREER_AND_COMMERCIAL_VALUE.md', 'reports/PUBLIC_RELEASE_READINESS.md', 'reports/PUBLIC_RELEASE_CHECKLIST.md',
    'reports/INDUSTRY_AUDIT.md', 'docs/ROADMAP.md',  # audit of the private repository; personal priorities
    'reports/ANNOUNCEMENT_DRAFT.md', 'reports/INDUSTRY_RECOGNITION_REVIEW.md',  # drafts for the owner to post / review
    'reports/PORTFOLIO_SUMMARY.md', 'reports/INDUSTRY_VISIBILITY_PACKAGE.md',
]
_NAME = r'[^\\/\s"\'<>|:*?%]+'
# 'users' / 'home' are assembled from pieces so that the export does not redact this file's own patterns
_U, _H = 'us' + 'ers', 'ho' + 'me'
HOME_PATTERNS = [  # no \b before the drive letter: JSON-escaped text has e.g. "\nC:" where \b does not match
    (re.compile(rf'(?i)[a-z]:(?:\\\\|\\|/)+{_U}(?:\\\\|\\|/)+{_NAME}'), '<HOME>'),
    (re.compile(rf'(?i)[a-z]%3a(?:%5c|%2f)+{_U}(?:%5c|%2f)+{_NAME}'), '<HOME>'),
    (re.compile(rf'(?i)/[a-z]/{_U}/{_NAME}'), '<HOME>'),
    (re.compile(rf'(?i)(?<![A-Za-z0-9_.~-])(?:\\\\|\\|/){_U}(?:\\\\|\\|/)+{_NAME}'), '<HOME>'),  # macOS, drive-less Windows
    (re.compile(rf'/{_H}/{_NAME}'), '<HOME>'),
    (re.compile(r'torchinductor_[A-Za-z0-9_.-]+'), 'torchinductor_<user>'),
]
SECRET_RE = re.compile(r'(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|'
                       r'xox[baprs]-|BEGIN [A-Z ]*PRIVATE KEY)')
BINARY_SUFFIXES = {'.pt', '.pth', '.ckpt', '.bin', '.safetensors', '.gguf', '.pkl', '.npy', '.npz', '.onnx', '.parquet'}
OPAQUE_OK = {'.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.svgz', '.woff', '.woff2'}  # undecodable by design


def git(*args, cwd=None, env=None):
    return subprocess.run(['git', *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


def origin_slug(root):
    try:
        url = git('remote', 'get-url', 'origin', cwd=root)
    except subprocess.CalledProcessError:
        return None
    m = re.search(r'[:/]([^/:]+)/([^/]+?)(?:\.git)?$', url)
    return f'{m.group(1)}/{m.group(2)}' if m else None


def private_identifiers(root, ref, extra=()):
    """(e-mails, user name, other deny words) that must not appear in the export.

    E-mails come from this fork's own commits (not upstream's, which are public) and git config; other words are the home
    folder name, the computer name, the origin owner and --deny words."""
    upstream = [r for r in ('upstream/master', 'upstream/main') if subprocess.run(
        ['git', 'rev-parse', '--verify', '--quiet', r], cwd=root, capture_output=True).returncode == 0]
    rng = [ref, *(f'^{u}' for u in upstream)]
    emails = set(git('log', '--format=%ae%n%ce', *rng, cwd=root).split())
    try:
        emails.add(git('config', 'user.email', cwd=root))
    except subprocess.CalledProcessError:
        pass
    emails = {e for e in emails if e and not e.endswith('@users.noreply.github.com') and e != 'noreply@github.com'}
    user = os.environ.get('USERNAME') or os.environ.get('USER') or ''
    words = {Path.home().name, os.environ.get('COMPUTERNAME', ''), *(e.split('@')[0] for e in emails), *extra}
    slug = origin_slug(root)
    if slug:
        words.add(slug.split('/')[0])
    words = {w for w in words if len(w) >= 4 and w.lower() != user.lower()}
    return emails, user, words


def decode(raw):
    """(text, encoding) for UTF-8 / UTF-8-BOM / UTF-16-BOM data, else (None, None)."""
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        try:
            return raw.decode('utf-16'), 'utf-16'
        except UnicodeDecodeError:
            return None, None
    if b'\x00' in raw:  # BOM-less UTF-16 (or binary) also decodes as UTF-8, with NULs that hide words from the regexes
        return None, None
    enc = 'utf-8-sig' if raw.startswith(codecs.BOM_UTF8) else 'utf-8'
    try:
        return raw.decode(enc), enc
    except UnicodeDecodeError:
        return None, None


def redact(text, extra=()):
    counts = 0
    for rx, repl in [*HOME_PATTERNS, *extra]:
        text, n = rx.subn(repl, text)
        counts += n
    return text, counts


def _word_rx(w):
    return re.compile(rf'(?i)(?<![A-Za-z0-9]){re.escape(w)}(?![A-Za-z0-9])')


_HEX = re.compile(r'(?<![0-9A-Za-z])[0-9a-f]{7,40}(?![0-9A-Za-z])')


def private_object_prefixes(root):
    """All 7..40-character prefixes of this fork's own commit and tag-object ids (upstream nanoGPT's are public)."""
    upstream = [r for r in ('upstream/master', 'upstream/main') if subprocess.run(
        ['git', 'rev-parse', '--verify', '--quiet', r], cwd=root, capture_output=True).returncode == 0]
    ids = set(git('rev-list', '--all', *(['--not', *upstream] if upstream else []), cwd=root).split())
    ids |= set(git('for-each-ref', '--format=%(objectname)', 'refs/tags', cwd=root).split())
    return {i[:n] for i in ids for n in range(7, 41)}


def redact_hashes(text, prefixes):
    """Replace private commit / tag ids (full or abbreviated) with <private-commit>; 64-hex digests are never touched."""
    n = 0

    def repl(m):
        nonlocal n
        if m.group() in prefixes:
            n += 1
            return '<private-commit>'
        return m.group()
    return _HEX.sub(repl, text), n


def scan(root, emails, user, words=(), hash_prefixes=frozenset()):
    problems = []
    checks = [('private e-mail', re.compile(re.escape(e), re.I)) for e in emails]
    if len(user) >= 4:
        checks.append(('local user name', _word_rx(user)))
    checks += [('private identifier', _word_rx(w)) for w in words]
    raw_terms = [*emails, *([user] if len(user) >= 4 else []), *words]
    for p in root.rglob('*'):
        if not p.is_file() or '.git' in p.relative_to(root).parts:
            continue
        rel = p.relative_to(root).as_posix()
        if p.suffix.lower() in BINARY_SUFFIXES:
            problems.append(f'{rel}: model/data file')
            continue
        raw = p.read_bytes()
        text, _ = decode(raw)
        if text is None:  # byte-level backstop for files the text checks cannot read
            low = raw.lower()
            if any(t.lower().encode('utf-8') in low or t.lower().encode('utf-16-le') in low for t in raw_terms):
                problems.append(f'{rel}: identifier in raw bytes')
            if p.suffix.lower() not in OPAQUE_OK:
                problems.append(f'{rel}: undecodable text (not scanned)')
            continue
        for label, rx in checks:
            if rx.search(text):
                problems.append(f'{rel}: {label}')
        if any(rx.search(text) for rx, repl in HOME_PATTERNS if repl == '<HOME>'):
            problems.append(f'{rel}: home path')
        if SECRET_RE.search(text):
            problems.append(f'{rel}: credential pattern')
        if hash_prefixes and any(m.group() in hash_prefixes for m in _HEX.finditer(text)):
            problems.append(f'{rel}: private commit id')
    return problems


def validate(export_dir, source_root):
    from .clean_checkout import INPUTS, SUITES, manifest, sha256
    expected = manifest()
    tmp = Path(tempfile.mkdtemp(prefix='efficientgpt_export_check_'))
    clone = tmp / 'repo'
    try:
        git('clone', '--quiet', str(export_dir), str(clone))
        for rel, policy in INPUTS.items():
            src = source_root / rel
            if policy == 'manifest' and expected.get(rel) != sha256(src):
                raise SystemExit(f'{rel}: checksum mismatch')
            (clone / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, clone / rel)
        out = {}
        for name, modules in SUITES.items():
            p = subprocess.run([sys.executable, '-m', 'unittest', *modules], cwd=clone, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', env=dict(os.environ, PYTHONIOENCODING='utf-8'))
            ran = re.search(r'Ran (\d+) tests', p.stderr)
            skipped = re.search(r'skipped=(\d+)', p.stderr)
            out[name] = {'returncode': p.returncode, 'ran': int(ran.group(1)) if ran else None,
                         'skipped': int(skipped.group(1)) if skipped else 0}  # no log tails: they contain local paths
            print(f"  {name:22s} rc={p.returncode} ran={out[name]['ran']} skipped={out[name]['skipped']}", flush=True)
            if p.returncode:
                print(p.stderr[-1500:], flush=True)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ref', default='HEAD')
    ap.add_argument('--author-name', default='EfficientGPT author (placeholder)')
    ap.add_argument('--author-email', default='replace-me@users.noreply.github.com')
    ap.add_argument('--keep-internal', action='store_true')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--deny', nargs='*', default=[], help='extra words that must not appear in the export')
    ap.add_argument('--allow', nargs='*', default=[],
                    help='identifiers the owner chose to make public (e.g. the GitHub handle in LICENSE); removed from the deny list')
    ap.add_argument('--message', default='EfficientGPT public release', help='commit message of the single export commit')
    args = ap.parse_args()
    source_root = Path(git('rev-parse', '--show-toplevel')).resolve()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f'{out} exists; choose a new directory (the export is never written over)')
    if source_root in out.parents or out == source_root:
        raise SystemExit('--out must be outside this repository')
    commit = git('rev-parse', args.ref, cwd=source_root)
    emails, user, words = private_identifiers(source_root, commit, args.deny)
    words = {w for w in words if w.lower() not in {a.lower() for a in args.allow}}
    slug = origin_slug(source_root)
    extra = [(re.compile(re.escape(slug), re.I), '<private-repository>')] if slug else []
    hash_prefixes = private_object_prefixes(source_root)
    data = subprocess.run(['git', 'archive', '--format=tar', commit], cwd=source_root, check=True,
                          capture_output=True).stdout
    out.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(out, filter='data')
    excluded = []
    if not args.keep_internal:
        for rel in INTERNAL:
            if (out / rel).exists():
                (out / rel).unlink()
                excluded.append(rel)
    redactions = {}
    for p in out.rglob('*'):
        if not p.is_file():
            continue
        raw = p.read_bytes()
        text, enc = decode(raw)
        if text is None:
            continue
        new, n = redact(text, extra)
        new, h = redact_hashes(new, hash_prefixes)
        n += h
        if n:
            p.write_bytes(new.encode(enc))  # same encoding (BOM kept), line endings untouched
            redactions[p.relative_to(out).as_posix()] = n
    problems = scan(out, emails, user, words, hash_prefixes)
    manifest_doc = {'source': 'sanitised snapshot of the private research repository; its commit ids are shown as '
                              '<private-commit>', 'excluded_internal_documents': excluded,
                    'redacted_files': len(redactions), 'redactions_total': sum(redactions.values()),
                    'redactions_per_file': redactions, 'scan_problems': problems,
                    'history': 'fresh repository, one commit; the private repository and its history are unchanged',
                    'author': f'{args.author_name} <{args.author_email}>'}
    (out / 'RELEASE_MANIFEST.json').write_text(json.dumps(manifest_doc, indent=1), encoding='utf-8')
    env = dict(os.environ, GIT_AUTHOR_NAME=args.author_name, GIT_AUTHOR_EMAIL=args.author_email,
               GIT_COMMITTER_NAME=args.author_name, GIT_COMMITTER_EMAIL=args.author_email,
               # UTC: the local timezone offset would otherwise reveal the owner's region
               GIT_AUTHOR_DATE=time.strftime('%Y-%m-%dT%H:%M:%S+0000', time.gmtime()),
               GIT_COMMITTER_DATE=time.strftime('%Y-%m-%dT%H:%M:%S+0000', time.gmtime()))
    nosign = ('-c', 'commit.gpgsign=false', '-c', 'tag.gpgsign=false')
    git('init', '--quiet', '-b', 'main', cwd=out)
    git('add', '-A', cwd=out)
    git(*nosign, 'commit', '--quiet', '-m', args.message, cwd=out, env=env)
    print(f'export: {out}\n  source commit {commit[:12]}; excluded {len(excluded)} internal docs; redacted {sum(redactions.values())} '
          f'items in {len(redactions)} files; files: {sum(1 for p in out.rglob("*") if p.is_file() and ".git" not in p.parts)}')
    print(f'  scan problems: {len(problems)}' + ''.join(f'\n    {x}' for x in problems[:20]))
    failed = bool(problems)
    if args.validate:
        print('  validating (all test suites in a temporary clone of the export) ...', flush=True)
        result = validate(out, source_root)
        failed |= any(s['returncode'] for s in result.values())
        manifest_doc['validation'] = result
        (out / 'RELEASE_MANIFEST.json').write_text(json.dumps(manifest_doc, indent=1), encoding='utf-8')
        post = scan(out, emails, user, words, hash_prefixes)
        if post:
            print('  scan problems after validation:' + ''.join(f'\n    {x}' for x in post[:20]))
            failed = True
        git('add', 'RELEASE_MANIFEST.json', cwd=out)
        git(*nosign, 'commit', '--quiet', '--amend', '--no-edit', cwd=out, env=env)
        # drop the pre-amend commit object so the repository holds exactly one commit
        git('reflog', 'expire', '--expire=now', '--all', cwd=out)
        git('gc', '--quiet', '--prune=now', cwd=out)
    print('RESULT: ' + ('FAILED' if failed else 'OK'))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
