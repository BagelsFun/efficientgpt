"""Verify complete/incomplete benchmark reports without inventing GPU results."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from .report import EXPECTED_CASES, EXPECTED_MODES, summarize


class TestReport(unittest.TestCase):
    def make_fake_files(self, folder, complete):
        path = Path(folder)
        results = path / 'rows.csv'
        memory = path / 'memory.csv'
        cases = sorted(EXPECTED_CASES) if complete else [(1, 8, 16)]
        rows=[]
        summ=[]
        for rnd in ([1, 2] if complete else [1]):
            for mode in EXPECTED_MODES:
                for batch, p, n in cases:
                    rows.append(dict(round=rnd,mode=mode,batch=batch,prompt_tokens=p,
                            generated_tokens=n,output_correctness='PASS',p50_ms=12.34))
                summ.append(dict(round=rnd,mode=mode,graphs_resident=0,
                     setup_peak_allocated_mib=10,steady_peak_allocated_mib=11,
                     steady_peak_reserved_mib=12,graph_capture_ms=0,
                     initialization_and_correctness_ms=13,correctness='PASS'))
        for filename, contents in [(results,rows),(memory,summ)]:
            with filename.open('w',newline='',encoding='utf8') as f:
                writer=csv.DictWriter(f,fieldnames=list(contents[0]))
                writer.writeheader()
                writer.writerows(contents)
        return results, memory

    def test_complete_matrix_accepted(self):
        with TemporaryDirectory() as tmp:
            files=self.make_fake_files(tmp, True)
            text,complete = summarize(*files)
            self.assertTrue(complete)
            self.assertIn('FULL PHASE 5 MATRIX VALIDATED',text)
            self.assertIn('12/12',text)

    def test_quick_matrix_is_partial(self):
        with TemporaryDirectory() as tmp:
            files=self.make_fake_files(tmp, False)
            text,complete = summarize(*files)
            self.assertFalse(complete)
            self.assertIn('PARTIAL / NOT YET FULLY VALIDATED',text)
            self.assertIn('1/12',text)

    def test_correctness_failure_invalidates_report(self):
        with TemporaryDirectory() as tmp:
            a,b=self.make_fake_files(tmp, True)
            s=a.read_text(encoding='utf8')
            a.write_text(s.replace('PASS','FAIL',1),encoding='utf8')
            text,complete = summarize(a,b)
            self.assertFalse(complete)
            self.assertIn('Correctness failure',text)


if __name__=='__main__':
    unittest.main()
