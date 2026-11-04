"""Styles for the read-only sequence crop heatmap in the DB record dialog."""

SEQUENCE_HEATMAP_CSS = """
.seq-heatmap{border:1px solid #e0e7f2;border-radius:12px;padding:20px;margin:0 0 22px;background:#fff;min-width:0}
.seq-heatmap [hidden]{display:none!important}.seq-heatmap h3{margin:0 0 6px;font-size:16px;letter-spacing:-.02em}.seq-heatmap h4{margin:0;font-size:13px;color:#314561}
.seq-heatmap .sh-help{font-size:11px;line-height:1.8;color:#718098;margin:5px 0 12px;overflow-wrap:anywhere}
.seq-heatmap .sh-controls{display:flex;align-items:end;justify-content:space-between;gap:14px;flex-wrap:wrap;margin:18px 0 14px}
.seq-heatmap .sh-map-field{display:flex;flex-direction:column;gap:6px;min-width:150px;max-width:100%;flex:1}
.seq-heatmap label{font-size:11px;font-weight:700;color:#61738c}.seq-heatmap select{border:1px solid #d6dfed;border-radius:8px;background:#fff;padding:9px 11px;color:#314561;max-width:100%;width:100%;font:inherit;font-size:12px}
.seq-heatmap .sh-tools{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.seq-heatmap .sh-segment{display:flex;gap:3px;border-radius:8px;background:#eef2f8;padding:3px}
.seq-heatmap .sh-segment button{font-size:11px;color:#78879c;background:transparent;padding:8px 11px;white-space:nowrap;border-radius:6px;font-weight:650}
.seq-heatmap .sh-segment button[aria-pressed=true]{background:white;color:#3157d5;box-shadow:0 1px 4px #20304e15}.seq-heatmap button:disabled{cursor:default;opacity:.45}
.seq-heatmap a{font-size:11px;color:#3157d5;text-decoration:none}.seq-heatmap a:hover{text-decoration:underline}.seq-heatmap :is(a,button,select,summary):focus-visible{outline:2px solid #6688ef;outline-offset:3px}
.seq-heatmap .sh-status{font-size:12px;color:#687b96;line-height:1.8;margin:12px 0}.seq-heatmap .sh-status.sh-error{color:#a43a3a;background:#fff3f1;border:1px solid #f2d7d2;border-radius:8px;padding:12px}
.seq-heatmap .sh-status button{font-size:11px;background:#edf2fb;color:#3157d5;margin-left:10px;padding:6px 10px}
.seq-heatmap figure{margin:0}.seq-heatmap .sh-stage{display:flex;justify-content:center;align-items:center;border:1px solid #e0e6f0;border-radius:9px;background:#f0f3f8;min-height:180px;overflow:hidden;padding:12px}
.seq-heatmap .sh-stage img{display:block;max-width:100%;max-height:450px;width:auto;height:auto;object-fit:contain}.seq-heatmap .sh-image-message{font-size:12px;text-align:center;color:#718098;padding:24px 12px;line-height:1.8;margin:0}
.seq-heatmap figcaption{font-size:10px;line-height:1.8;color:#74849c;margin-top:8px;overflow-wrap:anywhere}.seq-heatmap .sh-scale{max-width:490px;margin:13px auto 0}.seq-heatmap .sh-scale svg{display:block;width:100%;height:auto}
.seq-heatmap .sh-notes{background:#f6f8fc;border-radius:8px;padding:12px 14px;margin-top:15px;font-size:11px;line-height:1.9;color:#687b96;overflow-wrap:anywhere}.seq-heatmap .sh-notes p{margin:0}.seq-heatmap .sh-notes ul{margin:5px 0 0;padding-left:19px}
.seq-heatmap .sh-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:17px 0 22px}.seq-heatmap .sh-metric{min-width:0;background:#f6f8fc;border-radius:8px;padding:13px}
.seq-heatmap .sh-metric span{display:block;font-size:10px;color:#73839b;line-height:1.6}.seq-heatmap .sh-metric strong{display:block;font-size:20px;letter-spacing:-.03em;font-variant-numeric:tabular-nums;margin-top:6px;overflow-wrap:anywhere}
.seq-heatmap .sh-graphs{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:18px}.seq-heatmap .sh-chart-card{min-width:0}.seq-heatmap .sh-chart-scroll{overflow:auto;max-height:430px}.seq-heatmap .sh-chart-scroll>svg{display:block;width:100%;min-width:300px;height:auto}
.seq-heatmap .sh-empty{font-size:11px;line-height:1.8;color:#7a89a0;background:#f6f8fc;padding:25px 14px;border-radius:8px;text-align:center;margin:0}
.seq-heatmap .sh-data{margin-top:11px;font-size:11px}.seq-heatmap .sh-data summary{font-size:11px;color:#58709a;padding:5px 0}.seq-heatmap .sh-table{overflow:auto;max-height:230px;border:1px solid #e5ebf3;border-radius:7px;margin-top:6px}
.seq-heatmap .sh-table table{width:100%;font-size:10px}.seq-heatmap .sh-table :is(th,td){font-size:10px;padding:8px 10px}.seq-heatmap .sh-table th{position:sticky;top:0;background:#f1f5fb;z-index:1}.seq-heatmap .sh-footnote{margin-top:15px}
@media(max-width:760px){.seq-heatmap{padding:16px 14px}.seq-heatmap .sh-graphs{grid-template-columns:1fr}.seq-heatmap .sh-metric strong{font-size:17px}.seq-heatmap .sh-stage img{max-height:340px}.seq-heatmap .sh-tools{width:100%;justify-content:space-between}}
@media(max-width:460px){.seq-heatmap .sh-metrics{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.seq-heatmap .sh-metric{padding:10px}.seq-heatmap .sh-stage{min-height:150px;padding:7px}}
"""
