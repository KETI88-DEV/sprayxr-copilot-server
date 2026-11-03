"""One shared, scoped navigation bar for Swagger, database and motion pages."""
from __future__ import annotations

import re

_LINKS = (
    ("docs", "/docs", "API 문서", "Swagger API 문서"),
    ("database", "/admin", "DB 뷰어", "훈련 데이터와 세션 조회"),
    ("motion", "/motion-viewer", "3D 모션 뷰어", "전문가 모션과 작업 자세 비교"),
)

_CSS = """
:root{--copilot-nav-height:52px}
html{scroll-padding-top:calc(var(--copilot-nav-height) + 12px)}
#copilot-navigation,#copilot-navigation *{box-sizing:border-box}
#copilot-navigation{height:var(--copilot-nav-height);min-height:var(--copilot-nav-height);position:sticky;top:0;z-index:10000;display:flex;align-items:center;justify-content:space-between;gap:16px;padding:0 24px;background:#101d34;border-bottom:1px solid #293c5d;color:#eef4ff;box-shadow:0 2px 10px #0e1c3612;font-family:Inter,Pretendard,"Noto Sans KR",system-ui,sans-serif}
#copilot-navigation .portal-brand{display:flex;align-items:center;gap:9px;flex-shrink:0;color:#eef4ff;font-size:13px;font-weight:750;letter-spacing:.01em}
#copilot-navigation .portal-logo{display:grid;place-items:center;width:28px;height:28px;background:#24417a;border:1px solid #4566a2;border-radius:7px;font-size:10px;letter-spacing:.04em;color:#dce9ff}
#copilot-navigation .portal-links{display:flex;align-items:center;gap:5px;min-width:0}
#copilot-navigation .portal-links a{display:block;padding:8px 13px;border:1px solid transparent;border-radius:7px;text-decoration:none;color:#bbcbe5;font-size:12px;font-weight:650;line-height:18px;white-space:nowrap;transition:background .12s,color .12s}
#copilot-navigation .portal-links a:hover{background:#20324f;color:#fff}
#copilot-navigation .portal-links a[aria-current=page]{background:#3157d5;border-color:#5779e7;color:#fff}
#copilot-navigation .portal-links a:focus-visible{outline:2px solid #8cdaf0;outline-offset:2px}
body[data-portal-page=database] .app-shell{min-height:calc(100vh - var(--copilot-nav-height))}
body[data-portal-page=database] .sidebar{top:var(--copilot-nav-height);height:calc(100vh - var(--copilot-nav-height))}
body[data-portal-page=motion] .shell{height:calc(100dvh - var(--copilot-nav-height))}
body[data-portal-page=motion] .top{padding:14px 22px}
body[data-portal-page=motion] .top .title{font-size:22px;line-height:1.3;letter-spacing:-.025em}
body[data-portal-page=motion] .top .title small{font-size:12px;line-height:1.5;margin-top:5px;letter-spacing:0}
body[data-portal-page=database] .topbar h1{font-size:26px;line-height:1.3;letter-spacing:-.025em}
body[data-portal-page=database] .topbar .sub{line-height:1.6}
body[data-portal-page=docs]{margin:0}
body[data-portal-page=docs] .swagger-ui .info{margin:28px 0 24px}
body[data-portal-page=docs] .swagger-ui .info .title{font-size:26px;line-height:1.35;letter-spacing:-.025em;margin:0 0 14px}
body[data-portal-page=docs] .swagger-ui .info .description{font-size:14px;line-height:1.7}
body[data-portal-page=docs] .swagger-ui .info .title small{vertical-align:middle;letter-spacing:0}
body[data-portal-page=docs] .swagger-ui .opblock,body[data-portal-page=docs] .swagger-ui .opblock-tag{scroll-margin-top:calc(var(--copilot-nav-height) + 12px)}
@media(max-width:1200px){body[data-portal-page=motion] .shell{height:auto;min-height:calc(100dvh - var(--copilot-nav-height))}body[data-portal-page=motion] .stage{height:calc(100dvh - var(--copilot-nav-height) - 67px)}}
@media(max-width:850px){body[data-portal-page=database] .sidebar{top:auto;height:auto}}
@media(max-width:720px){body[data-portal-page=motion] .stage{height:calc(100dvh - var(--copilot-nav-height) - 86px)}}
@media(max-width:600px){:root{--copilot-nav-height:48px}#copilot-navigation{padding:0 12px;gap:9px}#copilot-navigation .portal-brand-name{display:none}#copilot-navigation .portal-links{gap:3px}#copilot-navigation .portal-links a{font-size:11px;padding:7px 9px}}
"""


def portal_document(document: str, active: str) -> str:
    """Insert a header without changing scripts, canvases, API URLs, or body attributes."""
    if 'id="copilot-navigation"' in document:
        return document
    if active not in {key for key, *_ in _LINKS}:
        raise ValueError("unknown portal page")
    links = "".join(
        f'<a data-portal-link="{key}" href="{path}" title="{title}"' +
        (' aria-current="page"' if key == active else '') + f'>{label}</a>'
        for key,path,label,title in _LINKS
    )
    bar = ('<header id="copilot-navigation"><div class="portal-brand">'
           '<span class="portal-logo" aria-hidden="true">XR</span><span class="portal-brand-name">AI Co-Pilot</span>'
           '</div><nav class="portal-links" aria-label="페이지 이동">' + links + '</nav></header>')
    document = document.replace('</head>', f'<style id="copilot-navigation-style">{_CSS}</style></head>', 1)
    return re.sub(r'<body\b[^>]*>', lambda match: match.group(0)[:-1] + f' data-portal-page="{active}">' + bar,
                  document, count=1, flags=re.IGNORECASE)
