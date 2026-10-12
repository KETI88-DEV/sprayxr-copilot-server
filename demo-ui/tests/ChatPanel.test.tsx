import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ChatPanel, type ChatMessage } from '../src/components/ChatPanel'

describe('ChatPanel', () => {
  it('submits on Enter and keeps Shift+Enter as a newline', async () => {
    const onSubmit = vi.fn()
    const onStop = vi.fn()
    const onReset = vi.fn()

    render(<ChatPanel messages={[]} busy={false} onSubmit={onSubmit} onStop={onStop} onReset={onReset} />)

    const input = screen.getByPlaceholderText('도장 관련 질문을 입력하세요')
    await userEvent.type(input, '첫 질문')
    await userEvent.keyboard('{Enter}')
    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith('첫 질문')
    })

    await userEvent.type(input, '다음 질문')
    await userEvent.keyboard('{Shift>}{Enter}{/Shift}')
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('scrolls to the bottom when messages grow', async () => {
    const onSubmit = vi.fn()
    const onStop = vi.fn()
    const onReset = vi.fn()
    const initialMessages: ChatMessage[] = [
      { role: 'user', text: '안녕' },
      { role: 'assistant', text: '반가워' },
    ]
    const { rerender } = render(
      <ChatPanel messages={initialMessages} busy={false} onSubmit={onSubmit} onStop={onStop} onReset={onReset} />,
    )

    const list = screen.getByRole('list')
    Object.defineProperty(list, 'clientHeight', { value: 120, configurable: true })
    Object.defineProperty(list, 'scrollHeight', { value: 120, configurable: true })
    Object.defineProperty(list, 'scrollTop', { value: 0, writable: true, configurable: true })

    Object.defineProperty(list, 'scrollHeight', { value: 360, configurable: true })
    rerender(
      <ChatPanel
        messages={[...initialMessages, { role: 'assistant', text: '추가 응답이 길어집니다' }]}
        busy={false}
        onSubmit={onSubmit}
        onStop={onStop}
        onReset={onReset}
      />,
    )

    await waitFor(() => {
      expect(list.scrollTop).toBe(240)
    })
  })
})

describe('ChatPanel 기능 버튼', () => {
  it('기능 4개 버튼을 그리고 누르면 그 task_type 을 넘긴다', () => {
    const calls: string[] = []
    render(
      <ChatPanel messages={[]} busy={false} onSubmit={() => {}} onStop={() => {}} onReset={() => {}} onPreset={(t) => calls.push(t)} />,
    )
    for (const taskType of ['knowledge_search', 'pose_recomm', 'trajectory_recomm', 'error_analysis']) {
      fireEvent.click(screen.getByTestId(`preset-${taskType}`))
    }
    expect(calls).toEqual(['knowledge_search', 'pose_recomm', 'trajectory_recomm', 'error_analysis'])
  })

  it('응답을 기다리는 동안은 누를 수 없다', () => {
    render(<ChatPanel messages={[]} busy onSubmit={() => {}} onStop={() => {}} onReset={() => {}} onPreset={() => {}} />)
    expect(screen.getByTestId('preset-pose_recomm')).toBeDisabled()
  })
})
