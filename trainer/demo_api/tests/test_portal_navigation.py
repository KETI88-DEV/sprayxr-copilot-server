from html.parser import HTMLParser
from fastapi.testclient import TestClient
from trainer.demo_api.app import create_app


class Navigation(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.inside=False
        self.count=0
        self.links=[]
        self.feed(document)
    def handle_starttag(self, tag, attrs):
        values=dict(attrs)
        if tag=='header' and values.get('id')=='copilot-navigation':
            self.inside=True
            self.count+=1
        if self.inside and tag=='a' and 'data-portal-link' in values:
            self.links.append((values.get('href'),values.get('aria-current')))
    def handle_endtag(self,tag):
        if tag=='header': self.inside=False


def test_all_three_pages_share_navigation_and_select_the_current_page(monkeypatch,tmp_path):
    monkeypatch.setenv('DEMO_LIVE_MODE','off')
    monkeypatch.setenv('SEQUENCE_DB_PATH',str(tmp_path/'nav.db'))
    client=TestClient(create_app())
    expected=['/docs','/admin','/motion-viewer']
    for path,selected in [('/docs','/docs'),('/admin','/admin'),('/admin/sessions','/admin'),('/motion-viewer','/motion-viewer')]:
        response=client.get(path)
        assert response.status_code==200
        nav=Navigation(response.text)
        assert nav.count==1
        assert [url for url,current in nav.links]==expected
        assert [url for url,current in nav.links if current=='page']==[selected]
        assert response.text.index('id="copilot-navigation"')<response.text.index('Home</h1>') if path=='/admin' else True
    docs=client.get('/docs').text
    assert 'SwaggerUIBundle' in docs and '/openapi.json' in docs
    assert client.get('/openapi.json').status_code==200
    assert client.get('/docs/oauth2-redirect').status_code==200
    assert client.get('/api/health').json()['status']=='ok'
