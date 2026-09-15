import sqlite3
import tempfile
import unittest
from pathlib import Path

from experiment.cctv_intelligence import cctv_tools


class IdentityProfileToolTests(unittest.TestCase):
    def test_missing_identity_tables_returns_setup_message(self) -> None:
        original_db_path = cctv_tools.DB_PATH
        try:
            with tempfile.TemporaryDirectory() as directory:
                cctv_tools.DB_PATH = Path(directory) / "test.sqlite3"
                conn = sqlite3.connect(cctv_tools.DB_PATH)
                try:
                    conn.execute("CREATE TABLE cameras (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
                    conn.execute("INSERT INTO cameras (id, name) VALUES (1, 'camera_1')")
                    conn.commit()
                finally:
                    conn.close()

                result = cctv_tools.get_identity_profile_summary("camera_1")

            self.assertIn("Run identity_profile_builder.py first", result)
        finally:
            cctv_tools.DB_PATH = original_db_path


if __name__ == "__main__":
    unittest.main()
