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
    expected=['/docs','/admin','/motion-viewer','https://github.com/KETI88-DEV/sprayxr-copilot-server']
    for path,selected in [('/docs','/docs'),('/admin','/admin'),('/admin/sessions','/admin'),('/admin/manage','/admin'),('/motion-viewer','/motion-viewer')]:
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
