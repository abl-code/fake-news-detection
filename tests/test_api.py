import unittest
from unittest.mock import patch
from backend.src.app import app


class ApiValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_invalid_json_types(self):
        for payload in ([], 'text', {'url': 42}, {'text': []}, {'lang': 'fr'}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post('/predict-text', json=payload).status_code, 400)

    def test_request_size_limit(self):
        self.assertEqual(self.client.post('/predict-text', json={'text': 'a' * 140000}).status_code, 413)

    def test_health_reports_missing_required_model(self):
        with patch('backend.src.app.MODELS', {}):
            self.assertEqual(self.client.get('/health').status_code, 503)

    def test_english_prediction(self):
        from backend.src.app import MODELS
        if 'en' not in MODELS:
            self.skipTest('No English artifacts')
        result = self.client.post('/predict-text', json={'text': 'Government officials announced a new public transport plan after several months of discussion with local community representatives.'})
        self.assertEqual(result.status_code, 200)
        self.assertIn(result.json['label'], ('REAL', 'FAKE'))


if __name__ == '__main__':
    unittest.main()
