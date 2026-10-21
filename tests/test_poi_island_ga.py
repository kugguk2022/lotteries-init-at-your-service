"""Fast isolated tests for PoI island GA; run: python -m unittest tests/test_poi_island_ga.py"""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE=Path(__file__).resolve().parents[1]/"experiments"/"poi_island_ga.py"
spec=importlib.util.spec_from_file_location("poi_island_ga", MODULE)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class IslandGATests(unittest.TestCase):
    def test_pool_validation_and_settlement(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"pools.csv"
            with p.open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=["draw_key","rank","main_numbers","auxiliary_numbers","score","created_utc"])
                w.writeheader()
                for n in range(1,5):
                    w.writerow({"draw_key":"2026-10-10","rank":n,"main_numbers":json.dumps([n,n+1,n+2,n+3,n+4]),
                                "auxiliary_numbers":"[1,2]","score":1/n,"created_utc":"2026-10-09T10:00:00Z"})
            pool=mod.read_pools(str(p),4)["2026-10-10"]
            self.assertEqual(len(pool),4)
            result=mod.settle([0],pool,((1,2,3,4,5),(1,2)),{"5+2":100.0},2.5)
            self.assertEqual(result["payout"],100.0)
            self.assertAlmostEqual(result["roi"],39.0)
            ga,diag=mod.evolve(pool,{},2,seed=42,islands=2,population=4,generations=3)
            self.assertEqual(len(set(ga)),2)
            self.assertEqual(len(diag),4)
    def test_late_pool_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"p.csv"
            p.write_text('draw_key,rank,main_numbers,auxiliary_numbers,score,created_utc\n2026-10-10,1,"[1,2,3,4,5]","[1,2]",1,2026-10-10T10:00:00Z\n')
            with self.assertRaises(ValueError):
                mod.read_pools(str(p),1)

if __name__=="__main__":
    unittest.main()
