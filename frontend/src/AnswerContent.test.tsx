import { render, screen } from '@testing-library/react'
import { expect, it } from 'vitest'
import AnswerContent from './AnswerContent'

it('renders model markdown and citations as accessible content', () => {
  render(<AnswerContent answer={'## Summary\n- **Artemis II** is crewed [S1, G1]\n- *Evidence grounded*'} />)
  expect(screen.getByRole('heading', { name: 'Summary' })).toBeInTheDocument()
  expect(screen.getByText('Artemis II').tagName).toBe('STRONG')
  expect(screen.getByText('[S1, G1]')).toHaveClass('citation-chip')
  expect(screen.getByText('Evidence grounded').tagName).toBe('EM')
})
