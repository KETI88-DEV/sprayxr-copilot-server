"""Shared portal navigation and a link to the source repository."""
from __future__ import annotations

import re

_LINKS = (
    ("docs", "/docs", "API 문서", "Swagger API 문서"),
    ("database", "/admin", "DB 뷰어", "훈련 데이터와 세션 조회"),
    ("motion", "/motion-viewer", "3D 모션 뷰어", "전문가 모션과 작업 자세 비교"),
    ("github", "https://github.com/KETI88-DEV/sprayxr-copilot-server", "GitHub", "GitHub 소스 저장소 · 새 탭에서 열기"),
)

_ICONS = {
    "docs": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z"/><path d="M14 3v6h6M8 13h8M8 17h6"/>',
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "motion": '<path d="m12 3 9 5v9l-9 5-9-5V8Zm0 10 9-5M3 8l9 5v9M7.5 5.5l9 5"/>',
    "github": '<circle cx="6" cy="5" r="3"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="19" r="3"/><path d="M6 8v8m0-4h6a6 6 0 0 0 6-6"/>',
}

_CSS = """
:root{--copilot-nav-height:52px}
html{scroll-padding-top:calc(var(--copilot-nav-height) + 12px)}
#copilot-navigation,#copilot-navigation *{box-sizing:border-box}
#copilot-navigation{height:var(--copilot-nav-height);min-height:var(--copilot-nav-height);position:sticky;top:0;z-index:10000;display:flex;align-items:center;justify-content:space-between;gap:16px;padding:0 24px;background:#101d34;border-bottom:1px solid #293c5d;color:#eef4ff;box-shadow:0 2px 10px #0e1c3612;font-family:Inter,Pretendard,"Noto Sans KR",system-ui,sans-serif}
#copilot-navigation .portal-brand{display:flex;align-items:center;gap:9px;flex-shrink:0;color:#eef4ff;font-size:13px;font-weight:750;letter-spacing:.01em}
#copilot-navigation .portal-logo{display:grid;place-items:center;width:28px;height:28px;background:#24417a;border:1px solid #4566a2;border-radius:7px;font-size:10px;letter-spacing:.04em;color:#dce9ff}
#copilot-navigation .portal-links{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));align-items:center;gap:6px;flex:0 1 560px;min-width:0}
#copilot-navigation .portal-links a{display:flex;align-items:center;justify-content:center;gap:7px;width:100%;height:36px;min-width:0;padding:8px 10px;border:1px solid transparent;border-radius:7px;text-decoration:none;color:#bbcbe5;font-size:12px;font-weight:650;line-height:18px;white-space:nowrap;transition:background .12s,color .12s}
#copilot-navigation .portal-icon{display:block;flex:0 0 18px;width:18px;height:18px}
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
@media(max-width:780px){#copilot-navigation{padding:0 12px;gap:9px}#copilot-navigation .portal-brand-name{display:none}#copilot-navigation .portal-links{flex:1}}
@media(max-width:720px){body[data-portal-page=motion] .stage{height:calc(100dvh - var(--copilot-nav-height) - 86px)}}
@media(max-width:600px){:root{--copilot-nav-height:48px}#copilot-navigation .portal-links{gap:4px}#copilot-navigation .portal-links a{font-size:11px;padding:6px;gap:5px;height:34px}#copilot-navigation .portal-icon{flex-basis:16px;width:16px;height:16px}}
@media(max-width:520px){:root{--copilot-nav-height:84px}#copilot-navigation .portal-links{grid-template-columns:repeat(2,minmax(0,1fr))}#copilot-navigation .portal-links a{height:32px}}
"""


def portal_document(document: str, active: str) -> str:
    """Insert a header without changing scripts, canvases, API URLs, or body attributes."""
    if 'id="copilot-navigation"' in document:
        return document
    if active not in {"docs", "database", "motion"}:
        raise ValueError("unknown portal page")
    links = "".join(
        f'<a data-portal-link="{key}" href="{path}" title="{title}"' +
        (' target="_blank" rel="noopener noreferrer" aria-label="GitHub 저장소 (새 탭)"' if key == "github" else '') +
        (' aria-current="page"' if key == active else '') +
        f'><svg class="portal-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">{_ICONS[key]}</svg><span>{label}</span></a>'
        for key,path,label,title in _LINKS
    )
    bar = ('<header id="copilot-navigation"><div class="portal-brand">'
           '<span class="portal-logo" aria-hidden="true">XR</span><span class="portal-brand-name">AI Co-Pilot</span>'
           '</div><nav class="portal-links" aria-label="페이지 이동">' + links + '</nav></header>')
    document = document.replace('</head>', f'<style id="copilot-navigation-style">{_CSS}</style></head>', 1)
    return re.sub(r'<body\b[^>]*>', lambda match: match.group(0)[:-1] + f' data-portal-page="{active}">' + bar,
                  document, count=1, flags=re.IGNORECASE)
