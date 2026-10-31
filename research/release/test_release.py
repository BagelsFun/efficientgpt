"""CPU tests for the public-export sanitiser (research/release/make_public_export.py).

Fixture paths are written with {U} / {H} placeholders (filled in by F) so that the export's own redaction leaves this
file intact and the tests still pass inside the exported tree."""
import tempfile
import unittest
from pathlib import Path

from .make_public_export import decode, redact, redact_hashes, scan


def F(s):
    return s.replace('{U}', 'Us' + 'ers').replace('{u}', 'us' + 'ers').replace('{H}', 'ho' + 'me').replace('{A}', 'alice')


class TestRedaction(unittest.TestCase):
    def test_home_paths_are_redacted(self):
        cases = {
            r'File "C:\{U}\alice\AI-Models\x.py", line 3': 'File "<HOME>\\AI-Models\\x.py", line 3',
            r'C:\\{U}\\alice\\AppData\\Local\\Temp': '<HOME>\\\\AppData\\\\Local\\\\Temp',
            'c:/{U}/alice/proj': '<HOME>/proj',
            '/c/{U}/alice/proj': '<HOME>/proj',
            '/{H}/alice/proj': '<HOME>/proj',
            'torchinductor_{A}/a5/x.py': 'torchinductor_<user>/a5/x.py',
            '/C/{u}/alice/proj': '<HOME>/proj',
            'see C%3A%5C{U}%5Calice%5Cx': 'see <HOME>%5Cx',
            '"/{U}/alice/proj"': '"<HOME>/proj"',
            r'at \{U}\alice\x': 'at <HOME>\\x',
            r'"line\nC:\\{U}\\alice\\x"': r'"line\n<HOME>\\x"',  # JSON-escaped newline right before the drive
        }
        for src, want in cases.items():
            got, n = redact(F(src))
            self.assertEqual(got, want, src)
            self.assertGreaterEqual(n, 1, src)

    def test_unrelated_text_untouched(self):
        for s in ('C:/Program Files/NVIDIA', '{u} of the model', '{U}/alice without drive', 'runs/20261031_v4/x.json',
                  'api/{u}/list'):
            self.assertEqual(redact(F(s)), (F(s), 0), s)

    def test_scan_fails_closed_on_other_encodings(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'ps.txt').write_bytes(F(r'owner Alice ran C:\{U}\alice').encode('utf-16'))  # BOM'd, like PowerShell `>`
            (root / 'nobom.log').write_bytes('BagelsCo alice'.encode('utf-16-le'))
            (root / 'pic.png').write_bytes(b'\x89PNG\xff\xfe')
            problems = '\n'.join(scan(root, set(), 'alice', {'BagelsCo'}))
            self.assertIn('ps.txt: local user name', problems)
            self.assertIn('ps.txt: home path', problems)
            self.assertIn('nobom.log: identifier in raw bytes', problems)
            self.assertIn('nobom.log: undecodable text', problems)
            self.assertNotIn('pic.png', problems)

    def test_redaction_preserves_utf16_and_crlf(self):
        text, enc = decode(F('a\r\nC:\\{U}\\alice\\x\r\n').encode('utf-16'))
        new, n = redact(text)
        self.assertEqual(n, 1)
        self.assertEqual(new.encode(enc).decode('utf-16'), 'a\r\n<HOME>\\x\r\n')

    def test_private_commit_ids_are_redacted(self):
        full = 'ab12cd3' + '4' * 33
        prefixes = {full[:n] for n in range(7, 41)}
        digest = 'ab12cd3' + '0' * 57  # a 64-hex SHA-256 starting with the same digits is not a commit id
        text = f'commit ab12cd3 + uncommitted; {{"commit": "{full}"}}; sha {digest}; ab12cd; upstream 3adf61e'
        got, n = redact_hashes(text, prefixes)
        self.assertEqual(n, 2)
        self.assertEqual(got, f'commit <private-commit> + uncommitted; {{"commit": "<private-commit>"}}; sha {digest}; '
                              'ab12cd; upstream 3adf61e')
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'a.json').write_text(text, encoding='utf-8')
            self.assertIn('a.json: private commit id', ' | '.join(scan(Path(d), set(), '', (), prefixes)))

    def test_scan_finds_leftovers(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'ok.md').write_text('clean text <HOME>/x', encoding='utf-8')
            (root / 'mail.md').write_text('contact a.person@example.com', encoding='utf-8')
            (root / 'path.log').write_text(F(r'C:\{U}\alice\x'), encoding='utf-8')
            (root / 'user.txt').write_text('owner Alice ran this', encoding='utf-8')
            (root / 'model.safetensors').write_bytes(b'\x00')
            (root / 'key.txt').write_text('token ghp_' + 'a' * 30, encoding='utf-8')
            problems = scan(root, {'a.person@example.com'}, 'alice')
            joined = '\n'.join(problems)
            for name in ('mail.md: private e-mail', 'path.log: home path', 'user.txt: local user name',
                         'model.safetensors: model/data file', 'key.txt: credential pattern'):
                self.assertIn(name, joined)
            self.assertNotIn('ok.md', joined)


if __name__ == '__main__':
    unittest.main()
