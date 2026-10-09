"""End-to-end tests on the simulator (no GPU, no video, no API key).

  cd experiment/cctv_intelligence && python -m unittest supervisor_agent.tests.test_supervisor_agent -v
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]          # cctv_intelligence/
TMP = Path(tempfile.mkdtemp(prefix="sa_test_"))
os.environ["CCTV_DB_PATH"] = str(TMP / "sim.sqlite3")
os.environ["SA_EXPORT_DIR"] = str(TMP / "exports")
os.environ["SA_LOG_DIR"] = str(TMP / "logs")

from supervisor_agent import db as D, derive, tools as T  # noqa: E402
from supervisor_agent.checker import unsupported  # noqa: E402


def setUpModule():
    subprocess.run([sys.executable, "-m", "supervisor_agent.sim.simulate", "--out", str(TMP), "--no-video"],
                   cwd=ROOT, check=True, capture_output=True)
    conn = D.connect()
    for r in D.camera_runs(conn, "cam07"):
        derive.build(conn, r["run_id"])


class DataLayer(unittest.TestCase):
    def setUp(self):
        self.conn = D.connect()
        self.today = "cam07_20261009_s1_sim"; self.yday = "cam07_20261008_s2_sim"

    def ev(self, run, typ):
        return self.conn.execute("SELECT start,end,zone FROM sa_events WHERE run_id=? AND type=?", (run, typ)).fetchall()

    def test_offline_detected(self):
        h = self.conn.execute("SELECT start,end FROM sa_health WHERE run_id=?", (self.today,)).fetchall()
        self.assertEqual([tuple(r) for r in h], [(345, 385)])

    def test_absence_matches_truth(self):
        ab = {r["zone"]: (r["start"], r["end"]) for r in self.ev(self.today, "absence")}
        self.assertIn("Op 15", ab)
        self.assertLess(abs(ab["Op 15"][0] - 200), 30); self.assertLess(abs(ab["Op 15"][1] - 461), 30)
        yz = {r["zone"] for r in self.ev(self.yday, "absence")}
        self.assertEqual(yz, {"Op 14", "Op 16"})

    def test_edge_counts_match_truth(self):
        for run in (self.today, self.yday):
            truth = json.loads((TMP / "truth" / f"{run}.json").read_text())
            n_in = self.conn.execute("SELECT count(*) FROM sa_edge_events WHERE run_id=? AND direction='in'", (run,)).fetchone()[0]
            self.assertLessEqual(abs(n_in - sum(truth["entries"].values())), 1)

    def test_new_detectors(self):
        self.assertTrue(self.ev(self.today, "fast_movement"))
        self.assertTrue(self.ev(self.today, "crowding"))
        ls = [r for r in self.ev(self.yday, "long_stationary") if "Notice board" in r["zone"] or "Bench R" in r["zone"]]
        self.assertTrue(ls)

    def test_counts_mostly_exact(self):
        truth = json.loads((TMP / "truth" / f"{self.yday}.json").read_text())["count_per_sec"]
        rows = self.conn.execute("SELECT start,end,value FROM sa_counts WHERE run_id=?", (self.yday,)).fetchall()
        ok = n = 0
        for s, e, v in rows:
            for t in range(s, e):
                n += 1; ok += (v == truth[t])
        self.assertGreater(ok / n, 0.9)


class Tools(unittest.TestCase):
    def setUp(self):
        self.conn = D.connect()

    def test_scene_log(self):
        r = T.get_scene_log(self.conn, "cam07", "2026-10-09 10:00", "10:10")
        self.assertIn("OFFLINE 10:05:45–10:06:25", r["text"])
        self.assertIn("Op 15", r["text"]); self.assertIn("EDGE CROSSINGS", r["text"])

    def test_place_resolver(self):
        r = T.get_camera_info(self.conn, "cam07", place="press 15 operator")
        self.assertIn("PLACE MATCH for 'press 15 operator': Op 15", r["text"])

    def test_positions_relation(self):
        r = T.get_scene_log(self.conn, "cam07", "2026-10-08 10:01:00", "10:02:30", detail="positions")
        self.assertIn("BETWEEN", r["text"])

    def test_overview_and_compare(self):
        r = T.get_dwell_overview(self.conn, "cam07", "2026-10-08 10:00", "10:10")
        self.assertIn("UNUSUAL", r["text"])
        c = T.compare_periods(self.conn, "cam07", "2026-10-08 10:00", "10:10", "2026-10-09 10:00", "10:10")
        self.assertIn("Op 15 empty_min", c["text"])

    def test_bad_window_is_actionable(self):
        r = T.call(self.conn, "get_scene_log", {"camera": "cam07", "start": "2026-10-01 10:00", "end": "10:10"})
        self.assertIn("Available recordings", r["text"])


class Checker(unittest.TestCase):
    def test_flags_invented(self):
        tool = ["Op 15 EMPTY 10:03:20–10:05:45", "absence 260 s"]
        self.assertEqual(unsupported("Op 15 was empty from 10:03:20 to 10:05:45 (about 4.3 min).", tool), [])
        self.assertIn("10:09:00", unsupported("Empty until 10:09:00, 37 people.", tool))
        self.assertIn("37", unsupported("Empty until 10:09:00, 37 people.", tool))


class FakeBlock(SimpleNamespace):
    def model_dump(self):
        return {k: v for k, v in self.__dict__.items()}


class AgentLoop(unittest.TestCase):
    def test_loop_with_fake_model(self):
        from supervisor_agent import agent as A
        calls = []

        class Msgs:
            def create(self, **kw):
                calls.append(kw)
                use = SimpleNamespace(input_tokens=10, output_tokens=5)
                if len(calls) == 1:
                    return SimpleNamespace(stop_reason="tool_use", usage=use, content=[FakeBlock(
                        type="tool_use", id="t1", name="get_scene_log",
                        input={"camera": "cam07", "start": "2026-10-09 10:04", "end": "10:07"})])
                return SimpleNamespace(stop_reason="end_turn", usage=use, content=[FakeBlock(
                    type="text", text="**Answer:** No. Op 15 was empty; camera offline 10:05:45–10:06:25.")])

        events = list(A.run_turn("Was Op 15 manned 10:04-10:07 today?", [], client=SimpleNamespace(messages=Msgs())))
        self.assertEqual([e["type"] for e in events][-2:], ["chunk", "done"])
        self.assertEqual(events[0]["name"], "get_scene_log")
        self.assertEqual(events[-1]["unsupported"], [])


if __name__ == "__main__":
    unittest.main()
