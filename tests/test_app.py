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
BOUNDARY = APP.parent / 'data/행정동경계/yeongcheon_dong_boundary.geojson'


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
            exists = Path.exists
            with patch.object(Path, 'exists', lambda p: False if p == BOUNDARY else exists(p)):
                app = self.start()
            self.assertEqual(len(app.dataframe[-1].value), len(pd.read_csv(SEED)))
            self.assertTrue(any('행정경계 파일이 없어' in x.value for x in app.info))
            self.assertFalse(self.path.exists())
        finally:
            os.chdir(previous)

    def test_bundled_boundary_and_grid(self):
        app = self.start()
        self.assertFalse(app.warning)
        self.assertFalse(any('행정경계 파일이 없어' in x.value for x in app.info))
        map_object = self.map_mock.call_args.args[0]
        layers = {getattr(child, 'layer_name', ''): child
                  for child in map_object._children.values()}
        self.assertEqual(len(layers['영천시 경계'].data['features']), 16)
        self.assertGreater(len(layers['탐색 격자 (0.0045도)'].data['features']), 0)

    def test_click_save_restart_and_duplicate(self):
        point = {'lat': 35.974912345, 'lng': 128.946112345}
        self.map_mock.return_value = {'last_clicked': point, 'center': point, 'zoom': 17}
        app = self.start()
        count = len(app.dataframe[-1].value)
        app.button[0].click().run(timeout=30)
        self.assertFalse(app.exception)
        saved = pd.read_csv(self.path)
        self.assertEqual(len(saved), count + 1)
        self.assertAlmostEqual(saved.iloc[-1]['위도'], point['lat'])
        self.assertIsNone(app.session_state['new_point'])
        restarted = self.start()
        self.assertEqual(len(restarted.dataframe[-1].value), count + 1)
        self.assertTrue(restarted.button(key='save_point').disabled)
        self.assertTrue(any('이미 저장된' in x.value for x in restarted.caption))

    def test_invalid_saved_csv_does_not_overwrite_it(self):
        self.path.write_text('wrong,columns\n1,2\n')
        app = self.start()
        self.assertTrue(app.error)
        self.assertEqual(self.path.read_text(), 'wrong,columns\n1,2\n')

    def test_undo_preserves_other_additions_and_can_save_again(self):
        point = {'lat': 35.97491234567891, 'lng': 128.94611234567891}
        self.map_mock.return_value = {'last_clicked': point}
        app = self.start()
        count = len(app.dataframe[-1].value)
        app.button(key='save_point').click().run(timeout=30)
        with self.path.open('a') as handle:
            handle.write('36.0,129.0\n')
        app.button(key='undo_point').click().run(timeout=30)
        self.assertFalse(app.error)
        saved = pd.read_csv(self.path)
        self.assertEqual(len(saved), count + 1)
        self.assertEqual(tuple(saved.iloc[-1]), (36.0, 129.0))
        self.assertTrue(app.button(key='undo_point').disabled)
        self.assertFalse(app.button(key='save_point').disabled)
        app.button(key='save_point').click().run(timeout=30)
        self.assertEqual(len(pd.read_csv(self.path)), count + 2)

    def test_failed_undo_keeps_saved_point(self):
        self.map_mock.return_value = {'last_clicked': {'lat': 35.97, 'lng': 128.94}}
        app = self.start()
        app.button(key='save_point').click().run(timeout=30)
        saved = self.path.read_bytes()
        with patch('pathlib.Path.replace', side_effect=PermissionError('read only')):
            app.button(key='undo_point').click().run(timeout=30)
        self.assertTrue(app.error)
        self.assertEqual(self.path.read_bytes(), saved)
        self.assertFalse(app.button(key='undo_point').disabled)

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
