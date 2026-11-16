(() => {
  'use strict';
  const state = document.getElementById('feedback-review-state');
  if (!state) return;
  let csrf;
  async function session() {
    try {
      const response = await fetch('/admin/api/manage/status', {credentials:'same-origin', cache:'no-store'});
      if (!response.ok) return;
      const data = await response.json();
      if (!data.authenticated || !data.csrf_token) return;
      csrf = data.csrf_token; state.textContent = `${data.username} 관리자 · 의견 처리 상태를 변경할 수 있습니다.`;
      document.querySelectorAll('[data-feedback-status],[data-feedback-save],[data-feedback-resolution]').forEach(node => {node.disabled = false;});
    } catch { state.textContent = '관리자 상태를 확인하지 못했습니다. 새로고침 후 다시 시도하세요.'; }
  }
  document.querySelectorAll('[data-feedback-save]').forEach(button => button.addEventListener('click', async () => {
    if (!csrf || button.disabled) return;
    const row = button.closest('tr'), select = row.querySelector('[data-feedback-status]'), result = row.querySelector('[data-feedback-result]');
    button.disabled = true; select.disabled = true; result.textContent = '저장 중…';
    try {
      const response = await fetch('/admin/api/manage/feedback-status', {method:'POST', credentials:'same-origin', headers:{'Content-Type':'application/json','X-DB-CSRF':csrf}, body:JSON.stringify({feedback_id:button.dataset.feedbackSave, status:select.value, resolution:row.querySelector('[data-feedback-resolution]').value})});
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) { csrf = null; state.textContent = '관리자 로그인이 만료되었습니다. 관리자 모드에서 다시 로그인하세요.'; }
        throw new Error('상태를 저장하지 못했습니다.');
      }
      const data = await response.json(); result.textContent = '저장됨 · ' + data.status_updated_at;
    } catch (error) { result.textContent = error.message || '연결을 확인하고 다시 시도하세요.'; }
    finally {button.disabled = !csrf; select.disabled = !csrf;}
  }));
  session();
})();
