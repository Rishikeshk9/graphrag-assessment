import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// Vitest only auto-registers React Testing Library cleanup when globals are on.
afterEach(() => {
  cleanup()
  localStorage.clear()
})

// jsdom implements neither layout nor scrolling.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {}
}
