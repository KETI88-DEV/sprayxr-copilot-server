from html.parser import HTMLParser
from fastapi.testclient import TestClient
from trainer.demo_api.app import create_app


class Navigation(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.inside=False
        self.count=0
        self.links=[]
        self.link_attributes=[]
        self.icons=[]
        self.current_link=None
        self.feed(document)
    def handle_starttag(self, tag, attrs):
        values=dict(attrs)
        if tag=='header' and values.get('id')=='copilot-navigation':
            self.inside=True
            self.count+=1
        if self.inside and tag=='a' and 'data-portal-link' in values:
            self.links.append((values.get('href'),values.get('aria-current')))
            self.link_attributes.append(values)
            self.current_link=values.get('href')
        if self.inside and tag=='svg' and self.current_link:
            assert values.get('aria-hidden')=='true'
            self.icons.append(self.current_link)
    def handle_endtag(self,tag):
        if tag=='header': self.inside=False
        if tag=='a': self.current_link=None


def test_all_three_pages_share_navigation_and_select_the_current_page(monkeypatch,tmp_path):
    monkeypatch.setenv('DEMO_LIVE_MODE','off')
    monkeypatch.setenv('SEQUENCE_DB_PATH',str(tmp_path/'nav.db'))
    client=TestClient(create_app())
    expected=['/home','/docs','/admin','/motion-viewer','/query-chat','https://github.com/KETI88-DEV/sprayxr-copilot-server']
    for path,selected in [('/home','/home'),('/docs','/docs'),('/admin','/admin'),('/admin/sessions','/admin'),('/admin/manage','/admin'),('/motion-viewer','/motion-viewer')]:
        response=client.get(path)
        assert response.status_code==200
        nav=Navigation(response.text)
        assert nav.count==1
        assert [url for url,current in nav.links]==expected
        assert nav.icons==expected
        assert [url for url,current in nav.links if current=='page']==[selected]
        external=nav.link_attributes[-1]
        assert external.get('target')=='_blank'
        assert {'noopener','noreferrer'} <= set(external.get('rel','').split())
        if path == '/admin':
            assert response.text.index('id="copilot-navigation"') < response.text.index('<h1>훈련 데이터 대시보드</h1>')
    docs=client.get('/docs').text
    assert 'SwaggerUIBundle' in docs and '/openapi.json' in docs
    assert client.get('/openapi.json').status_code==200
    assert client.get('/docs/oauth2-redirect').status_code==200
    assert client.get('/api/health').json()['status']=='ok'


def test_query_qa_update_shows_real_answers_with_verified_or_expected_corrections(monkeypatch, tmp_path):
    monkeypatch.setenv('DEMO_LIVE_MODE', 'off')
    monkeypatch.setenv('SEQUENCE_DB_PATH', str(tmp_path / 'home-qa.db'))
    client = TestClient(create_app())
    assert 'id="release-query-qa"' in client.get('/home').text

    posts = client.get('/home/api/updates').json()['updates']
    recent = [row for row in posts if row['version'] in ('v3.15.10', 'v3.15.11')]
    assert len(recent) == 2
    assert all(row['has_query_qa'] is True and 'query_qa' not in row for row in recent)

    detail = client.get('/home/api/updates/v3.15.11').json()['query_qa']
    questions = {row['id']: row for row in detail['questions']}
    assert len(questions) == 100
    assert all(row['question'] and row['baseline']['answer'] for row in questions.values())
    assert sum('retest' in row for row in questions.values()) == 43
    assert questions['D08']['baseline']['verdict'] == 'fail'
    assert '85%' in questions['D08']['baseline']['answer']
    assert questions['D08']['retest']['verdict'] == 'pass'
    assert '분사 기록' in questions['D08']['retest']['answer']
    assert questions['W07']['retest']['verdict'] == 'fail'
    assert '처짐' in questions['W07']['expected_answer']
    assert questions['W33']['baseline']['verdict'] == 'fail' and 'retest' not in questions['W33']
    assert questions['W33']['expected_answer']
    assert 'query_qa' not in client.get('/home/api/updates/v3.15.9').json()


def test_followup_qa_preserves_historical_passes(monkeypatch, tmp_path):
    monkeypatch.setenv('DEMO_LIVE_MODE', 'off')
    monkeypatch.setenv('SEQUENCE_DB_PATH', str(tmp_path / 'home-followup.db'))
    client = TestClient(create_app())

    response = client.get('/home/api/updates/v3.15.12')
    assert response.is_success
    detail = response.json()['query_qa']
    followups = [row for row in detail['questions'] if 'followup' in row]
    assert len(followups) == 44
    assert all(row['baseline']['verdict'] != 'pass' and row.get('retest', {}).get('verdict') != 'pass'
               for row in followups)
    assert detail['followup_version'] == 'v3.15.12'
    assert detail['followup_counts'] == {'pass': 44, 'partial': 0, 'fail': 0}
    assert all(row['followup']['http'] == 200 and row['followup']['verdict'] == 'pass' for row in followups)
    assert '.qa-answer.qa-answer-pass' in client.get('/assets/portal_home.css').text


def test_new_qa_campaign_uses_distinct_report(monkeypatch, tmp_path):
    monkeypatch.setenv('DEMO_LIVE_MODE', 'off')
    monkeypatch.setenv('SEQUENCE_DB_PATH', str(tmp_path / 'home-round2.db'))
    client = TestClient(create_app())

    first = client.get('/home/api/updates/v3.15.12').json()['query_qa']
    second = client.get('/home/api/updates/v3.15.13').json()['query_qa']
    assert len(first['questions']) == len(second['questions']) == 100
    assert {row['question'] for row in first['questions']}.isdisjoint(
        row['question'] for row in second['questions']
    )
    assert sum(second['baseline_counts'].values()) == 100
    pending = [row for row in second['questions'] if row['baseline']['verdict'] != 'pass']
    assert all(row['retest']['answer'] for row in pending)
    assert sum(second['retest_counts'].values()) == len(pending)
