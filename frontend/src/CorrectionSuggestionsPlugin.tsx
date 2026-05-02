import { useCallback } from 'react'
import { $createParagraphNode, $createTextNode, $getRoot } from 'lexical'
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext'

export type CorrectionSuggestion = {
  id: string
  start: number
  end: number
  replacement: string
  reason: 'grammar' | 'typo' | 'punctuation' | 'clarity'
}

type CorrectionSuggestionsPluginProps = {
  suggestions: CorrectionSuggestion[]
  onAccept: (suggestionId: string, nextText: string) => void
  onDismiss: (suggestionId: string) => void
}

function getPlainTextFromRoot() {
  return $getRoot().getTextContent()
}

function replaceTextRange(text: string, start: number, end: number, replacement: string) {
  return `${text.slice(0, start)}${replacement}${text.slice(end)}`
}

function setPlainText(text: string) {
  const root = $getRoot()
  root.clear()

  const paragraphs = text.split('\n')
  for (const paragraphText of paragraphs) {
    const paragraph = $createParagraphNode()
    paragraph.append($createTextNode(paragraphText))
    root.append(paragraph)
  }
}

export function CorrectionSuggestionsPlugin({
  suggestions,
  onAccept,
  onDismiss,
}: CorrectionSuggestionsPluginProps) {
  const [editor] = useLexicalComposerContext()

  const handleAccept = useCallback(
    (suggestion: CorrectionSuggestion) => {
      let nextText = ''
      editor.update(() => {
        const currentText = getPlainTextFromRoot()
        if (
          suggestion.start < 0 ||
          suggestion.end <= suggestion.start ||
          suggestion.end > currentText.length
        ) {
          return
        }
        nextText = replaceTextRange(
          currentText,
          suggestion.start,
          suggestion.end,
          suggestion.replacement,
        )
        setPlainText(nextText)
      })

      if (nextText) {
        onAccept(suggestion.id, nextText)
      }
    },
    [editor, onAccept],
  )

  if (suggestions.length === 0) {
    return null
  }

  return (
    <aside className="correction-panel" aria-label="Correction suggestions">
      <div className="correction-panel-header">
        <span>Corrections</span>
        <span>{suggestions.length}</span>
      </div>
      <div className="correction-list">
        {suggestions.map((suggestion) => (
          <div className="correction-item" key={suggestion.id}>
            <div className="correction-copy">
              <span className="correction-reason">{suggestion.reason}</span>
              <span className="correction-replacement">{suggestion.replacement}</span>
            </div>
            <div className="correction-actions">
              <button type="button" onClick={() => handleAccept(suggestion)}>
                Accept
              </button>
              <button type="button" onClick={() => onDismiss(suggestion.id)}>
                Dismiss
              </button>
            </div>
          </div>
        ))}
      </div>
    </aside>
  )
}
