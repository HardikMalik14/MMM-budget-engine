import base64
import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from index import app
from models.mmm import ChannelParams, MarketingMixModel, MMMConfig

client = TestClient(app)


def upload(body, name='weekly.csv'):
    return {'source': 'upload', 'file_name': name, 'file_data': base64.b64encode(body).decode()}


def test_home_and_health():
    assert 'Make every channel count.' in client.get('/').text
    assert client.get('/api/health').json()['status'] == 'ok'


def test_default_analysis_matches_original_python_model():
    response = client.post('/api/analyze', json={})
    assert response.status_code == 200, response.text
    report = response.json()
    data = pd.read_csv('data/synthetic_mmm_data.csv')
    config = MMMConfig(target_col='total_revenue', channel_cols=report['channels'],
                       control_cols=report['controls'], date_col='date')
    model = MarketingMixModel(config).fit(data, auto_tune=True, n_passes=2)
    np.testing.assert_allclose([r['fitted'] for r in report['decomposition']], model.predict())
    allocation = report['optimization']['allocation']
    budget = report['optimization']['total_budget']
    np.testing.assert_allclose(sum(r['optimal_spend'] for r in allocation), budget, rtol=1e-8)
    for row in allocation:
        assert row['lower_bound'] - 0.01 <= row['optimal_spend'] <= row['upper_bound'] + 0.01


def test_uploaded_csv_and_xlsx_are_processed_without_server_side_session():
    data = pd.read_csv('data/synthetic_mmm_data.csv')
    for name, payload in [('weekly.csv', data.to_csv(index=False).encode())]:
        options = upload(payload, name)
        schema = client.post('/api/schema', json=options)
        assert schema.status_code == 200
        response = client.post('/api/analyze', json={**options, 'auto_tune': False})
        assert response.status_code == 200, response.text
        assert response.json()['source'] == name
    buffer = io.BytesIO()
    data.to_excel(buffer, index=False)
    response = client.post('/api/schema', json=upload(buffer.getvalue(), 'weekly.xlsx'))
    assert response.status_code == 200, response.text
    assert response.json()['rows'] == len(data)


def test_infeasible_constraints_are_never_silently_relaxed():
    options = {'auto_tune': False, 'min_share': .4, 'max_share': .5}
    response = client.post('/api/analyze', json=options)
    assert response.status_code == 400
    assert 'infeasible' in response.json()['detail'].lower()
    response = client.post('/api/analyze', json={**options, 'auto_relax': True})
    assert response.status_code == 200
    assert response.json()['optimization']['relaxations']


def test_validation_and_upload_limits():
    assert client.post('/api/analyze', json={'weeks': 10000}).status_code == 422
    assert client.post('/api/analyze', json={'budget': -4}).status_code == 422
    assert client.post('/api/schema', json={'source': 'upload', 'file_name': 'x.csv', 'file_data': '???'}).status_code == 400
    assert client.post('/api/analyze', content=b'x' * 3_000_001).status_code == 413
    assert client.post('/api/analyze', json={'channels': ['total_revenue']}).status_code == 400


def test_changed_synthetic_settings_do_not_write_generated_data_to_disk():
    from pathlib import Path
    path = Path('data/synthetic_mmm_data.csv')
    original = path.read_bytes()
    response = client.post('/api/analyze', json={'seed': 17, 'weeks': 104, 'auto_tune': False})
    assert response.status_code == 200, response.text
    assert len(response.json()['data']) == 104
    assert path.read_bytes() == original
