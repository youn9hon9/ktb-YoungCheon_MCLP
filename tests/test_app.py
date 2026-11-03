"""Run with: .venv/bin/python -m unittest discover -s tests"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app.py'
SEED = APP.parent / 'data/영천시/영천시_횡단보도_좌표.csv'


class CollectionAppTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.path = Path(self.directory.name) / 'saved.csv'
        self.env = patch.dict(os.environ, {'CROSSWALK_CSV_PATH': str(self.path)})
        self.env.start()
        self.map = patch('streamlit_folium.st_folium', return_value={})
        self.map_mock = self.map.start()
        self.seed_bytes = SEED.read_bytes()

    def tearDown(self):
        self.assertEqual(SEED.read_bytes(), self.seed_bytes)
        self.map.stop()
        self.env.stop()
        self.directory.cleanup()

    def start(self):
        app = AppTest.from_file(str(APP)).run(timeout=30)
        self.assertFalse(app.exception)
        return app

    def test_missing_boundary_and_different_working_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.directory.name)
            app = self.start()
            self.assertEqual(len(app.dataframe[0].value), len(pd.read_csv(SEED)))
            self.assertTrue(any('행정경계 파일이 없어' in x.value for x in app.info))
            self.assertFalse(self.path.exists())
        finally:
            os.chdir(previous)

    def test_click_save_restart_and_duplicate(self):
        point = {'lat': 35.974912345, 'lng': 128.946112345}
        self.map_mock.return_value = {'last_clicked': point, 'center': point, 'zoom': 17}
        app = self.start()
        count = len(app.dataframe[0].value)
        app.button[0].click().run(timeout=30)
        self.assertFalse(app.exception)
        saved = pd.read_csv(self.path)
        self.assertEqual(len(saved), count + 1)
        self.assertAlmostEqual(saved.iloc[-1]['위도'], point['lat'])
        self.assertIsNone(app.session_state['new_point'])
        restarted = self.start()
        self.assertEqual(len(restarted.dataframe[0].value), count + 1)
        self.assertEqual(len(restarted.button), 0)
        self.assertTrue(any('이미 저장된' in x.value for x in restarted.info))

    def test_invalid_saved_csv_does_not_overwrite_it(self):
        self.path.write_text('wrong,columns\n1,2\n')
        app = self.start()
        self.assertTrue(app.error)
        self.assertEqual(self.path.read_text(), 'wrong,columns\n1,2\n')

    def test_failed_save_retains_pending_point(self):
        self.map_mock.return_value = {'last_clicked': {'lat': 35.97, 'lng': 128.94}}
        app = self.start()
        with patch('pathlib.Path.replace', side_effect=PermissionError('read only')):
            app.button[0].click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertEqual(app.session_state['new_point'], (35.97, 128.94))
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.glob('*.csv')), [])


if __name__ == '__main__':
    unittest.main()
