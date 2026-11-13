(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const open = byId('copilot-feedback-open'), dialog = byId('copilot-feedback-dialog');
  if (!open || !dialog) return;
  const form = byId('copilot-feedback-form'), message = byId('copilot-feedback-message');
  const name = byId('copilot-feedback-name'), kind = byId('copilot-feedback-kind');
  const submit = byId('copilot-feedback-submit'), reset = byId('copilot-feedback-reset');
  const state = byId('copilot-feedback-state');
  let pending = false, submissionId;
  function uuid() {
    if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
    const bytes = new Uint8Array(16);
    if (globalThis.crypto?.getRandomValues) crypto.getRandomValues(bytes);
    else for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
    bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
    const hex = Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
  }
  function notice(text = '', result = '') { state.textContent = text; state.dataset.state = result; }
  function clear() { form.reset(); submissionId = uuid(); notice(); message.focus(); }
  open.addEventListener('click', () => {
    if (!dialog.open) dialog.showModal();
    if (!submissionId) submissionId = uuid();
    message.focus();
  });
  byId('copilot-feedback-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => open.focus());
  // Native dialog handles Escape and keyboard focus trapping.
  reset.addEventListener('click', () => { if (!pending) clear(); });
  form.addEventListener('input', () => { if (!pending) { submissionId = uuid(); notice(); } });
  form.addEventListener('change', () => { if (!pending) { submissionId = uuid(); notice(); } });
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (pending) return;
    if (!message.value.trim()) { notice('의견 내용을 입력하세요.', 'error'); message.focus(); return; }
    if (!form.reportValidity()) return;
    pending = true; submit.disabled = true; reset.disabled = true;
    message.readOnly = true; name.readOnly = true; kind.disabled = true;
    form.setAttribute('aria-busy', 'true'); submit.textContent = '보내는 중…'; notice('의견을 저장하고 있습니다.');
    const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch('/api/feedback', {
        method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
        body: JSON.stringify({submission_id: submissionId || (submissionId = uuid()), message: message.value.trim(), name: name.value.trim(), kind: kind.value, page: document.body.dataset.portalPage}),
        signal: controller.signal,
      });
      if (!response.ok) {
        const details = {413:'입력이 너무 깁니다. 의견은 4,000자 이내로 입력하세요.', 422:'의견과 이름의 길이를 확인하세요.', 409:'전송 번호가 겹쳤습니다. 입력을 확인하고 다시 보내주세요.', 403:'현재 사이트의 의견 창에서 다시 보내주세요.'};
        if (response.status === 409) submissionId = uuid();
        throw new Error(details[response.status] || '저장하지 못했습니다. 입력은 유지됩니다. 잠시 후 다시 보내주세요.');
      }
      const receipt = await response.json();
      if (receipt.status !== 'received' || !receipt.feedback_id) throw new Error('접수 확인을 받지 못했습니다. 입력을 유지한 채 다시 보내주세요.');
      form.reset(); submissionId = uuid(); notice('의견이 접수되었습니다. 감사합니다.', 'success');
    } catch (error) {
      notice(error.name === 'AbortError' ? '접수 확인이 지연됩니다. 입력은 유지됩니다. 다시 보내면 중복 접수되지 않습니다.' : error.name === 'Error' && error.message ? error.message : '연결을 확인하고 다시 보내주세요. 입력은 유지됩니다.', 'error');
    } finally {
      clearTimeout(timeout); pending = false; submit.disabled = false; reset.disabled = false;
      message.readOnly = false; name.readOnly = false; kind.disabled = false;
      form.removeAttribute('aria-busy'); submit.textContent = '보내기';
      if (dialog.open) { if (state.dataset.state === 'success') reset.focus(); else message.focus(); }
    }
  });
})();
