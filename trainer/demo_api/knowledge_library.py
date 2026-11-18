"""Browse the indexed text; serve only explicitly registered original media."""
from __future__ import annotations
import hashlib,json,os,re
from functools import lru_cache
from pathlib import Path,PurePosixPath
import yaml
from fastapi import APIRouter,HTTPException,Query
from fastapi.responses import FileResponse,Response

_ID=re.compile(r'^[A-Za-z0-9_.:-]{1,160}$')
_STATIC=Path(__file__).with_name('static')
_POLICY_PATH=Path(__file__).resolve().parents[1]/'configs'/'evidence_exclusions.json'
_CORRECTIONS_PATH=Path(__file__).resolve().parents[1]/'configs'/'evidence_corrections.json'
_EXCLUDED_TEXT='확인된 오류 또는 손상으로 검색·학습 안내에서 제외된 문단입니다. 본문은 표시하지 않습니다.'

def _stamp(path: Path):
    try:
        stat=path.stat()
        return (str(path.resolve()),stat.st_mtime_ns,stat.st_size,stat.st_ctime_ns,stat.st_ino)
    except OSError:
        return None

def _metadata_path() -> Path | None:
    path=Path(os.getenv('CONFIG_PATH',''))
    if not path.is_file() or path.name.startswith('.env') or path.stat().st_size>1024*1024:
        return None
    try:
        config=yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        value=config.get('paths',{}).get('metadata_json')
        candidate=Path(value) if isinstance(value,str) else None
        return candidate if candidate and not candidate.name.startswith('.env') else None
    except (OSError,ValueError,AttributeError,TypeError,yaml.YAMLError):
        return None

@lru_cache(maxsize=2)
def _exclusion_policy(stamp):
    """Read only the authoritative ID/category policy, without retrieval imports."""
    try:
        if stamp is None or stamp[2]>1024*1024:
            raise ValueError('policy unavailable')
        data=json.loads(Path(stamp[0]).read_text(encoding='utf-8'))
        rows=data.get('조각') if isinstance(data,dict) else None
        if not isinstance(rows,dict):
            raise ValueError('invalid policy')
        excluded={}
        for chunk_id,reason in rows.items():
            if not isinstance(chunk_id,str) or not _ID.fullmatch(chunk_id) or not isinstance(reason,str) or not reason.strip():
                raise ValueError('invalid policy entry')
            # Full reasons can repeat the incorrect instruction. Publish the category only.
            category,separator,_=reason.partition(':')
            excluded[chunk_id]=category.strip()[:80] if separator and category.strip() else '검토 제외'
        return excluded
    except (OSError,ValueError,TypeError):
        raise HTTPException(503,'자료 검토 정책을 확인할 수 없어 학습 자료 조회를 잠시 중단합니다.') from None


@lru_cache(maxsize=2)
def _correction_policy(stamp):
    """Load corrected bodies only; never expose source/audit claims from the policy."""
    try:
        if stamp is None or stamp[2]>1024*1024:
            raise ValueError('policy unavailable')
        data=json.loads(Path(stamp[0]).read_text(encoding='utf-8'))
        rows=data.get('조각') if isinstance(data,dict) else None
        if not isinstance(rows,dict):
            raise ValueError('invalid policy')
        corrected={}
        for chunk_id,row in rows.items():
            if not isinstance(chunk_id,str) or not _ID.fullmatch(chunk_id) or not isinstance(row,dict):
                raise ValueError('invalid policy entry')
            text=row.get('본문')
            if not isinstance(text,str) or not text.strip() or len(text)>64*1024:
                raise ValueError('invalid corrected body')
            corrected[chunk_id]=text
        return corrected
    except (OSError,ValueError,TypeError):
        raise HTTPException(503,'자료 교정 정책을 확인할 수 없어 학습 자료 조회를 잠시 중단합니다.') from None


@lru_cache(maxsize=2)
def _indexed(stamp,policy_stamp,corrections_stamp):
    excluded=_exclusion_policy(policy_stamp)
    corrections=_correction_policy(corrections_stamp)
    docs={};chunks={}
    if stamp is None or stamp[2]>256*1024*1024:
        return docs,chunks
    try:
        data=json.loads(Path(stamp[0]).read_text(encoding='utf-8'))
        rows=data.get('id_map',[])
        rows=rows.values() if isinstance(rows,dict) else rows
        for row in rows:
            if not isinstance(row,dict):
                continue
            doc_id=row.get('doc_id');chunk_id=row.get('chunk_id');text=row.get('text')
            if not all(isinstance(x,str) and x for x in (doc_id,chunk_id,text)):
                continue
            if not _ID.fullmatch(doc_id) or not _ID.fullmatch(chunk_id):
                continue
            reason=excluded.get(chunk_id)
            corrected=chunk_id in corrections and not reason
            effective_text=_EXCLUDED_TEXT if reason else corrections.get(chunk_id,text)
            chunk={'chunk_id':chunk_id,'doc_id':doc_id,'text':effective_text,
                   'page':None if reason else row.get('page'),'excluded':bool(reason),'exclusion_reason':reason,
                   'corrected':bool(corrected),'correction_basis':'원본 대조 교정본' if corrected else None}
            chunks[chunk_id]=chunk
            if doc_id not in docs:
                docs[doc_id]={'doc_id':doc_id,'title':doc_id,'title_source':'문서 ID',
                              'source_type':str(row.get('source_type','text')),'chunks':[]}
            docs[doc_id]['chunks'].append(chunk)
        for doc in docs.values():
            doc['chunks'].sort(key=lambda row:row['chunk_id'])
            allowed=[chunk for chunk in doc['chunks'] if not chunk['excluded']]
            doc['excluded_count']=len(doc['chunks'])-len(allowed)
            doc['corrected_count']=sum(chunk['corrected'] for chunk in allowed)
            if allowed:
                first=next((line.strip().lstrip('# ').strip() for line in allowed[0]['text'].splitlines() if line.strip()),doc['doc_id'])
                doc['title']=first[:100]
                doc['title_source']='원본 대조 교정본 첫 문장' if allowed[0]['corrected'] else '본문 첫 문장'
            doc['search']=' '.join(row['text'] for row in allowed).casefold()
    except (OSError,ValueError,TypeError):
        return {},{}
    return docs,chunks

def _media_root():
    return Path(os.getenv('EXPERT_REFERENCE_ROOT','/expert_reference_data')).resolve()/'knowledge_library'

def _registered_media():
    root=_media_root();manifest=root/'manifest.json'
    registrations={};assets={}
    if root.is_symlink() or root.resolve()!=root:
        return registrations,assets
    try:
        if manifest.is_symlink() or manifest.stat().st_size>2*1024*1024:
            return registrations,assets
        data=json.loads(manifest.read_text(encoding='utf-8')).get('documents',{})
        for doc_id,row in data.items():
            if not _ID.fullmatch(doc_id) or not isinstance(row,dict):
                continue
            item={'title':str(row.get('title',''))[:200],'assets':[]}
            for raw in row.get('assets',[]):
                if not isinstance(raw,dict) or not isinstance(raw.get('file'),str):
                    continue
                rel=PurePosixPath(raw['file'])
                if rel.is_absolute() or '..' in rel.parts or '\\' in raw['file'] or any(part.startswith('.env') for part in rel.parts):
                    continue
                path=root/Path(*rel.parts)
                if path.is_symlink() or path.resolve()!=path or not path.is_file() or not path.is_relative_to(root):
                    continue
                if path.stat().st_size>50*1024*1024:
                    continue
                with path.open('rb') as stream:
                    head=stream.read(16)
                extension=path.suffix.lower()
                mime=('application/pdf' if extension=='.pdf' and head.startswith(b'%PDF-') else
                      'image/png' if extension=='.png' and head.startswith(b'\x89PNG\r\n\x1a\n') else
                      'image/jpeg' if extension in {'.jpg','.jpeg'} and head.startswith(b'\xff\xd8\xff') else
                      'image/webp' if extension=='.webp' and head[:4]==b'RIFF' and head[8:12]==b'WEBP' else None)
                if mime is None:
                    continue
                asset_id=hashlib.sha256((doc_id+'\0'+rel.as_posix()).encode()).hexdigest()[:32]
                public={'id':asset_id,'kind':'pdf' if mime=='application/pdf' else 'image',
                        'label':str(raw.get('label') or path.name)[:200],
                        'url':'/query-chat/api/knowledge/files/'+asset_id}
                item['assets'].append(public);assets[asset_id]=(path,mime)
            registrations[doc_id]=item
    except (OSError,ValueError,TypeError):
        return {},{}
    return registrations,assets

def _public(doc,registered):
    assets=registered.get('assets',[])
    return {'doc_id':doc['doc_id'],'title':registered.get('title') or doc['title'],
            'title_source':'등록 제목' if registered.get('title') else doc['title_source'],
            'source_type':doc['source_type'],'chunk_count':len(doc['chunks']),'excluded_count':doc['excluded_count'],
            'corrected_count':doc['corrected_count'],
            'original_available':bool(assets),'image_count':sum(a['kind']=='image' for a in assets),
            'assets':assets,'text_basis':'검색용 텍스트 문단'+(' · 원본 대조 교정본 포함' if doc['corrected_count'] else '')+' · 원문 페이지 배치와 다를 수 있음',
            'original_basis':'별도 등록된 원본 PDF·이미지 · 검색용 교정 텍스트와 구분'}

def create_knowledge_library_router():
    router=APIRouter(include_in_schema=False)
    def content():
        path=_metadata_path()
        return _indexed(_stamp(path) if path else None,_stamp(_POLICY_PATH),_stamp(_CORRECTIONS_PATH))

    @router.get('/query-chat/api/knowledge/documents')
    def documents(q: str=Query('',max_length=200),offset: int=Query(0,ge=0),limit: int=Query(50,ge=1,le=100)):
        docs,_=content();registered,_=_registered_media();needle=q.strip().casefold()
        rows=[_public(doc,registered.get(key,{})) for key,doc in docs.items()
              if not needle or needle in doc['search'] or needle in key.casefold() or needle in registered.get(key,{}).get('title','').casefold()]
        rows.sort(key=lambda row:(row['title'],row['doc_id']))
        return {'documents':rows[offset:offset+limit],'total':len(rows),'offset':offset,
                'images_available':any(row['image_count'] for row in rows),
                'note':'검색에 사용하는 텍스트를 문서별로 표시합니다. 원본 PDF·이미지는 별도로 연결된 자료에만 표시됩니다.'}

    @router.get('/query-chat/api/knowledge/documents/{doc_id}')
    def document(doc_id: str,offset: int=Query(0,ge=0),limit: int=Query(20,ge=1,le=100)):
        docs,_=content();doc=docs.get(doc_id)
        if doc is None:
            raise HTTPException(404,'등록된 학습 자료를 찾을 수 없습니다.')
        registered,_=_registered_media()
        return {**_public(doc,registered.get(doc_id,{})),'chunks':doc['chunks'][offset:offset+limit],'offset':offset,
                'next_offset':offset+limit if offset+limit<len(doc['chunks']) else None}

    @router.get('/query-chat/api/knowledge/chunks/{chunk_id}')
    def chunk(chunk_id: str):
        _,chunks=content()
        if chunk_id not in chunks:
            raise HTTPException(404,'답변의 근거 문단을 찾을 수 없습니다.')
        return chunks[chunk_id]

    @router.get('/query-chat/api/knowledge/documents/{doc_id}/text')
    def document_text(doc_id: str):
        docs,_=content();doc=docs.get(doc_id)
        if doc is None:
            raise HTTPException(404,'등록된 학습 자료를 찾을 수 없습니다.')
        paragraphs=[('[제외 문단 · '+row['exclusion_reason']+']\n' if row['excluded'] else '[원본 대조 교정본]\n' if row['corrected'] else '')+row['text'] for row in doc['chunks']]
        text='검색용 텍스트 문단 재구성본입니다. 원문 페이지 배치 및 이미지와 다를 수 있습니다.\n원본 대조 교정본은 별도로 표시하며 등록된 원본 PDF·이미지와 구분합니다.\n확인된 오류 또는 손상으로 제외된 문단은 본문을 표시하지 않습니다.\n\n'+'\n\n'.join(paragraphs)
        return Response(text,media_type='text/plain',headers={'Content-Disposition':'attachment; filename="'+doc_id+'.txt"'})

    @router.get('/query-chat/api/knowledge/files/{asset_id}')
    def original(asset_id: str):
        _exclusion_policy(_stamp(_POLICY_PATH))
        _correction_policy(_stamp(_CORRECTIONS_PATH))
        _,assets=_registered_media()
        if asset_id not in assets:
            raise HTTPException(404,'연결된 원본 자료를 찾을 수 없습니다.')
        path,mime=assets[asset_id]
        filename=path.name if all(ord(c)>=32 and ord(c)!=127 for c in path.name) else 'document'+path.suffix
        return FileResponse(path,media_type=mime,filename=filename,content_disposition_type='inline',
                            headers={'X-Content-Type-Options':'nosniff','Cache-Control':'private, max-age=60'})

    def static_asset(filename: str,content_type: str):
        def asset():
            return FileResponse(_STATIC/filename,media_type=content_type,headers={'Cache-Control':'no-cache'})
        return asset

    for name,mime in [('knowledge_library.js','text/javascript'),('knowledge_library.css','text/css')]:
        router.add_api_route('/assets/'+name,static_asset(name,mime),methods=['GET'])
    return router
