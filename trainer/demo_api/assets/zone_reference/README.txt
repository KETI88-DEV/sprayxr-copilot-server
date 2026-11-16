Zone 영역 자료 (TSB, 2026-10-15)

1. zone_rects.csv — Zone 별 경계 사각형 (2번 컬러 라벨맵 2048 x 2048 기준)
   model_key    : 피도면 모델 (Learn01~Learn10 학습 · Exam2 자격2 · UnitX-Y Unit 전용)
   map_num      : 면 번호 (= 높이맵 파일명 MapNN). Zone 번호는 면마다 1부터 다시 시작
   zone_id      : Zone 번호 (= analyze-sequence 의 zone_id)
   rect_x, rect_y, rect_w, rect_h : 높이맵 PNG 기준 Rect, 0~1 정규화, 원점 = 이미지 좌상단
   px_x, px_y, px_w, px_h         : 같은 Rect 의 2048 x 2048 픽셀 좌표
   u_min, v_min, u_max, v_max     : 같은 영역의 텍스처 UV (원점 = 좌하단)
   area_percent : 텍스처 전체 대비 Zone 실제 면적(%)
   ※ Zone 은 사각형이 아니어서 일부 Rect 는 이웃 Zone 과 크게 겹칩니다(최대 약 50%). 정밀 구분은 2번 라벨맵을 사용해 주십시오.

2. color/<model_key>_<map_num>.png — 컬러 라벨맵 (2048 x 2048, Zone 설계 원본)
   픽셀 색 = Zone (색 ↔ zone_id 는 palette.csv) · 검정 또는 투명 = Zone 아님
   팔레트 외 색 없음(전 파일 검사 완료) · 높이맵 PNG 와 같은 방향(최상단 행 = v=1)
   높이맵 해상도가 다르면 최근접(nearest) 방식으로 맞춰 주십시오.

3. palette.csv — zone_id 1~39 ↔ RGB

4. scenario_map.csv — scenario_id → model_key
   학습: scenario_id 1~10 = Learn01~Learn10 · Unit: scenario_id "Unit<번호>"
   Unit 은 학습·자격2 모델을 재사용하는 경우가 많아 같은 라벨맵을 공유합니다. model_key 가 비어 있으면 Zone 정의가 없는 Unit 입니다.
   실전(Practice)·검정은 Zone 미적용입니다.
