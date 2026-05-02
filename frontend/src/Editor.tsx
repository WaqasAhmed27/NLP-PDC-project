import { useCallback, useEffect, useRef, useState } from 'react'
import { $getRoot, $getSelection, $isElementNode, $isRangeSelection, $isTextNode, type ElementNode, type LexicalNode } from 'lexical'
import { LexicalComposer } from '@lexical/react/LexicalComposer'
import { PlainTextPlugin } from '@lexical/react/LexicalPlainTextPlugin'
import { ContentEditable } from '@lexical/react/LexicalContentEditable'
import { HistoryPlugin } from '@lexical/react/LexicalHistoryPlugin'
import { OnChangePlugin } from '@lexical/react/LexicalOnChangePlugin'
import { LexicalErrorBoundary } from '@lexical/react/LexicalErrorBoundary'
import { useEditorSocket, type IncomingEditorMessage } from './useEditorSocket'
import { AutocompleteNode, $isAutocompleteNode } from './AutocompleteNode'
import { AutocompletePlugin, type TokenChunkEvent } from './AutocompletePlugin'
import {
  CorrectionSuggestionsPlugin,
  type CorrectionSuggestion,
} from './CorrectionSuggestionsPlugin'
import { FloatingToolbarPlugin } from './FloatingToolbarPlugin'

type PendingEdit = {
  text: string
  cursorCharIndex: number
  shouldAutocomplete: boolean
  shouldCorrect: boolean
}

type DocumentStats = {
  words: number
  characters: number
}

type ActivityEntry = {
  id: number
  label: string
  detail: string
  time: string
}

const EDIT_DEBOUNCE_MS = 50
const AUTOCOMPLETE_DEBOUNCE_MS = 250
const CORRECTION_DEBOUNCE_MS = 1200
const MAX_ACTIVITY_ENTRIES = 6

const lexicalTheme = {
  paragraph: 'editor-paragraph',
}

function getNodeTextLength(node: LexicalNode) {
  return getGhostFreeTextContent(node).length
}

function getGhostFreeTextContent(node: LexicalNode): string {
  if ($isAutocompleteNode(node)) {
    return ''
  }

  if (!$isElementNode(node)) {
    return node.getTextContent()
  }

  const children = node.getChildren()
  const isRoot = node.getParent() === null

  return children
    .map((child) => getGhostFreeTextContent(child))
    .join(isRoot ? '\n' : '')
}

function getElementTextOffset(element: ElementNode, childOffset: number) {
  return element
    .getChildren()
    .slice(0, childOffset)
    .reduce((total, child) => total + getNodeTextLength(child), 0)
}

function getOffsetInsideNode(node: LexicalNode, offset: number) {
  if ($isTextNode(node)) {
    return offset
  }

  if ($isElementNode(node)) {
    return getElementTextOffset(node, offset)
  }

  return 0
}

function getAbsoluteOffsetInPlainText(
  currentNode: LexicalNode,
  targetNodeKey: string,
  targetLocalOffset: number,
): number | null {
  if (currentNode.getKey() === targetNodeKey) {
    return targetLocalOffset
  }

  if (!$isElementNode(currentNode)) {
    return null
  }

  let runningOffset = 0
  const children = currentNode.getChildren()
  const isRoot = currentNode.getParent() === null

  for (let index = 0; index < children.length; index += 1) {
    const child = children[index]
    const foundOffset = getAbsoluteOffsetInPlainText(
      child,
      targetNodeKey,
      targetLocalOffset,
    )

    if (foundOffset !== null) {
      return runningOffset + foundOffset
    }

    runningOffset += getNodeTextLength(child)

    if (isRoot && index < children.length - 1) {
      runningOffset += 1
    }
  }

  return null
}

function getCursorCharacterIndex() {
  const root = $getRoot()
  const selection = $getSelection()
  const text = getGhostFreeTextContent(root)

  if (!$isRangeSelection(selection)) {
    return text.length
  }

  const focus = selection.focus
  const focusNode = focus.getNode()
  const localOffset = getOffsetInsideNode(focusNode, focus.offset)
  const absoluteOffset = getAbsoluteOffsetInPlainText(
    root,
    focusNode.getKey(),
    localOffset,
  )

  return absoluteOffset ?? text.length
}

function getFirstChangedCharacterIndex(previousText: string, nextText: string) {
  const maxSharedLength = Math.min(previousText.length, nextText.length)

  for (let index = 0; index < maxSharedLength; index += 1) {
    if (previousText[index] !== nextText[index]) {
      return index
    }
  }

  return maxSharedLength
}

function getDocumentStats(text: string): DocumentStats {
  const trimmedText = text.trim()

  return {
    words: trimmedText ? trimmedText.split(/\s+/).length : 0,
    characters: text.length,
  }
}

function formatActivityTime() {
  return new Date().toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function summarizeChunk(chunk: string) {
  const normalized = chunk.replace(/\s+/g, ' ').trim()

  if (!normalized) {
    return ''
  }

  return normalized.length > 72 ? `${normalized.slice(0, 72)}...` : normalized
}

function getMessageSummary(payload: IncomingEditorMessage) {
  if (payload.type === 'token') {
    return payload.chunk ? `token: "${summarizeChunk(payload.chunk)}"` : 'token'
  }

  if (payload.type === 'corrections') {
    return 'corrections received'
  }

  if (payload.type === 'done') {
    return 'generation complete'
  }

  if (payload.type === 'cancelled') {
    return 'generation cancelled'
  }

  if (payload.type === 'server_error') {
    return payload.chunk ? `error: ${summarizeChunk(payload.chunk)}` : 'server error'
  }

  return payload.type
}

export function Editor() {
  const debounceTimerRef = useRef<number | null>(null)
  const autocompleteTimerRef = useRef<number | null>(null)
  const correctionTimerRef = useRef<number | null>(null)
  const pendingEditRef = useRef<PendingEdit | null>(null)
  const lastSentTextRef = useRef('')
  const lastSentCursorRef = useRef(0)
  const lastCorrectionTextRef = useRef('')
  const latestCorrectionRequestIdRef = useRef<string | null>(null)
  const suppressIncomingTokensRef = useRef(false)
  const suppressEditorSyncRef = useRef(false)
  const activeRewriteRequestIdRef = useRef<string | null>(null)
  const tokenChunkIdRef = useRef(0)
  const rewriteChunkIdRef = useRef(0)
  const rewriteDoneIdRef = useRef(0)
  const activityIdRef = useRef(0)
  const [lastMessage, setLastMessage] = useState<IncomingEditorMessage | null>(null)
  const [tokenChunk, setTokenChunk] = useState<TokenChunkEvent | null>(null)
  const [rewriteChunks, setRewriteChunks] = useState<TokenChunkEvent[]>([])
  const [rewriteDoneId, setRewriteDoneId] = useState(0)
  const [correctionSuggestions, setCorrectionSuggestions] = useState<CorrectionSuggestion[]>([])
  const [documentStats, setDocumentStats] = useState<DocumentStats>({
    words: 0,
    characters: 0,
  })
  const [activityLog, setActivityLog] = useState<ActivityEntry[]>([])
  const [isAutocompleteEnabled, setIsAutocompleteEnabled] = useState(true)
  const [isRewriteEnabled, setIsRewriteEnabled] = useState(true)
  const [isCorrectionEnabled, setIsCorrectionEnabled] = useState(true)

  const appendActivity = useCallback((label: string, detail: string) => {
    activityIdRef.current += 1
    const entry = {
      id: activityIdRef.current,
      label,
      detail,
      time: formatActivityTime(),
    }

    setActivityLog((entries) => [entry, ...entries].slice(0, MAX_ACTIVITY_ENTRIES))
  }, [])

  const handleSocketMessage = useCallback((payload: IncomingEditorMessage) => {
    setLastMessage(payload)
    appendActivity(payload.type, getMessageSummary(payload))

    if (payload.request_id === activeRewriteRequestIdRef.current) {
      if (payload.type === 'token' && payload.chunk) {
        rewriteChunkIdRef.current += 1
        const rewriteChunk = {
          id: rewriteChunkIdRef.current,
          chunk: payload.chunk,
        }
        setRewriteChunks((chunks) => [...chunks, rewriteChunk])
      }

      if (
        payload.type === 'done' ||
        payload.type === 'cancelled' ||
        payload.type === 'server_error'
      ) {
        rewriteDoneIdRef.current += 1
        setRewriteDoneId(rewriteDoneIdRef.current)
        activeRewriteRequestIdRef.current = null
        suppressEditorSyncRef.current = false
      }

      console.info('editor socket rewrite message', payload)
      return
    }

    if (payload.type === 'corrections') {
      if (!isCorrectionEnabled) {
        return
      }

      try {
        const parsed = JSON.parse(payload.chunk) as Omit<CorrectionSuggestion, 'id'>[]
        if (payload.request_id !== latestCorrectionRequestIdRef.current) {
          console.info('accepting out-of-order corrections payload', {
            request_id: payload.request_id,
            latest_correction_request_id: latestCorrectionRequestIdRef.current,
            count: parsed.length,
          })
        }
        setCorrectionSuggestions(
          parsed.map((suggestion, index) => ({
            ...suggestion,
            id: `${payload.request_id}-${index}`,
          })),
        )
      } catch (error) {
        console.error('Failed to parse correction suggestions', error)
      }
      return
    }

    if (payload.type === 'token' && payload.chunk && !suppressIncomingTokensRef.current) {
      tokenChunkIdRef.current += 1
      setTokenChunk({
        id: tokenChunkIdRef.current,
        chunk: payload.chunk,
      })
    }

    console.info('editor socket message', payload)
  }, [appendActivity, isCorrectionEnabled])

  const { status, sendMessage, sendRewriteRequest } = useEditorSocket({
    onMessage: handleSocketMessage,
  })

  useEffect(() => {
    return () => {
      if (debounceTimerRef.current !== null) {
        window.clearTimeout(debounceTimerRef.current)
      }

      if (autocompleteTimerRef.current !== null) {
        window.clearTimeout(autocompleteTimerRef.current)
      }

      if (correctionTimerRef.current !== null) {
        window.clearTimeout(correctionTimerRef.current)
      }
    }
  }, [])

  const scheduleAutocomplete = useCallback(
    (text: string, cursorCharIndex: number) => {
      if (autocompleteTimerRef.current !== null) {
        window.clearTimeout(autocompleteTimerRef.current)
      }

      if (!isAutocompleteEnabled || !text.trim()) {
        return
      }

      autocompleteTimerRef.current = window.setTimeout(() => {
        autocompleteTimerRef.current = null
        sendMessage({
          action: 'autocomplete',
          newText: text,
          editCharIndex: cursorCharIndex,
        })
        suppressIncomingTokensRef.current = false
      }, AUTOCOMPLETE_DEBOUNCE_MS)
    },
    [isAutocompleteEnabled, sendMessage],
  )

  const scheduleCorrection = useCallback(
    (text: string, cursorCharIndex: number) => {
      if (correctionTimerRef.current !== null) {
        window.clearTimeout(correctionTimerRef.current)
      }

      if (!isCorrectionEnabled || !text.trim() || text === lastCorrectionTextRef.current) {
        return
      }

      correctionTimerRef.current = window.setTimeout(() => {
        correctionTimerRef.current = null
        const requestId = sendMessage({
          action: 'correct',
          newText: text,
          editCharIndex: cursorCharIndex,
        })
        if (requestId) {
          latestCorrectionRequestIdRef.current = requestId
          lastCorrectionTextRef.current = text
        }
      }, CORRECTION_DEBOUNCE_MS)
    },
    [isCorrectionEnabled, sendMessage],
  )

  const handleGhostDismiss = useCallback(() => {
    suppressIncomingTokensRef.current = true
    setTokenChunk(null)
  }, [])

  const handleRewriteRequest = useCallback(
    (highlightedText: string, instruction: string) => {
      const requestId = sendRewriteRequest({
        highlightedText,
        instruction,
      })

      if (requestId) {
        activeRewriteRequestIdRef.current = requestId
        suppressEditorSyncRef.current = true
        suppressIncomingTokensRef.current = true
        pendingEditRef.current = null
        if (debounceTimerRef.current !== null) {
          window.clearTimeout(debounceTimerRef.current)
          debounceTimerRef.current = null
        }
        if (autocompleteTimerRef.current !== null) {
          window.clearTimeout(autocompleteTimerRef.current)
          autocompleteTimerRef.current = null
        }
        if (correctionTimerRef.current !== null) {
          window.clearTimeout(correctionTimerRef.current)
          correctionTimerRef.current = null
        }
        setTokenChunk(null)
        setRewriteChunks([])
        setCorrectionSuggestions([])
      }

      return requestId
    },
    [sendRewriteRequest],
  )

  const handleAutocompleteToggle = useCallback(
    (enabled: boolean) => {
      setIsAutocompleteEnabled(enabled)

      if (enabled) {
        return
      }

      if (autocompleteTimerRef.current !== null) {
        window.clearTimeout(autocompleteTimerRef.current)
        autocompleteTimerRef.current = null
      }

      handleGhostDismiss()
    },
    [handleGhostDismiss],
  )

  const handleCorrectionToggle = useCallback((enabled: boolean) => {
    setIsCorrectionEnabled(enabled)

    if (enabled) {
      return
    }

    if (correctionTimerRef.current !== null) {
      window.clearTimeout(correctionTimerRef.current)
      correctionTimerRef.current = null
    }

    latestCorrectionRequestIdRef.current = null
    setCorrectionSuggestions([])
  }, [])

  const flushPendingEdit = useCallback(() => {
    const pendingEdit = pendingEditRef.current
    pendingEditRef.current = null

    if (!pendingEdit) {
      return
    }

    const lastSentText = lastSentTextRef.current
    const lastSentCursor = lastSentCursorRef.current
    if (
      lastSentText === pendingEdit.text &&
      lastSentCursor === pendingEdit.cursorCharIndex
    ) {
      return
    }

    const textChanged = lastSentText !== pendingEdit.text
    const editCharIndex = textChanged
      ? getFirstChangedCharacterIndex(lastSentText, pendingEdit.text)
      : pendingEdit.cursorCharIndex
    const safeEditCharIndex = Math.min(editCharIndex, lastSentText.length)

    const sent = sendMessage({
      action: 'edit',
      newText: pendingEdit.text,
      editCharIndex: safeEditCharIndex,
    })

    if (sent) {
      lastSentTextRef.current = pendingEdit.text
      lastSentCursorRef.current = pendingEdit.cursorCharIndex
      console.info('sent editor edit', {
        new_text: pendingEdit.text,
        edit_char_index: safeEditCharIndex,
        cursor_char_index: pendingEdit.cursorCharIndex,
      })
      if (pendingEdit.shouldAutocomplete) {
        scheduleAutocomplete(pendingEdit.text, pendingEdit.cursorCharIndex)
      }
      if (pendingEdit.shouldCorrect) {
        scheduleCorrection(pendingEdit.text, pendingEdit.cursorCharIndex)
      }
    }
  }, [scheduleAutocomplete, scheduleCorrection, sendMessage])

  const handleCorrectionAccept = useCallback((suggestionId: string, nextText: string) => {
    setCorrectionSuggestions((suggestions) =>
      suggestions.filter((suggestion) => suggestion.id !== suggestionId),
    )
    lastCorrectionTextRef.current = nextText
  }, [])

  const handleCorrectionDismiss = useCallback((suggestionId: string) => {
    setCorrectionSuggestions((suggestions) =>
      suggestions.filter((suggestion) => suggestion.id !== suggestionId),
    )
  }, [])

  const scheduleEdit = useCallback(
    (edit: PendingEdit) => {
      pendingEditRef.current = edit
      if (edit.text !== lastSentTextRef.current) {
        setCorrectionSuggestions([])
      }

      if (debounceTimerRef.current !== null) {
        window.clearTimeout(debounceTimerRef.current)
      }

      if (autocompleteTimerRef.current !== null) {
        window.clearTimeout(autocompleteTimerRef.current)
      }

      if (correctionTimerRef.current !== null) {
        window.clearTimeout(correctionTimerRef.current)
      }

      debounceTimerRef.current = window.setTimeout(() => {
        debounceTimerRef.current = null
        flushPendingEdit()
      }, EDIT_DEBOUNCE_MS)
    },
    [flushPendingEdit],
  )

  const initialConfig = {
    namespace: 'Phase5Editor',
    theme: lexicalTheme,
    nodes: [AutocompleteNode],
    onError(error: Error) {
      throw error
    },
  }

  return (
    <main className="editor-shell">
      <LexicalComposer initialConfig={initialConfig}>
        <section className="editor-workspace" aria-label="Realtime editor workspace">
          <div className="editor-card">
            <div className="editor-toolbar" aria-label="Editor controls">
              <div className="toolbar-status-group">
                <span className={`status-dot status-dot-${status}`} aria-hidden="true" />
                <span className="toolbar-status-label">{status}</span>
                <span className="toolbar-divider" aria-hidden="true" />
                <span className="toolbar-meta">{documentStats.words} words</span>
                <span className="toolbar-meta">{documentStats.characters} chars</span>
              </div>

              <div className="toolbar-control-group">
                <label className="feature-switch">
                  <input
                    type="checkbox"
                    checked={isAutocompleteEnabled}
                    onChange={(event) => handleAutocompleteToggle(event.target.checked)}
                  />
                  <span>Autocomplete</span>
                </label>
                <label className="feature-switch">
                  <input
                    type="checkbox"
                    checked={isCorrectionEnabled}
                    onChange={(event) => handleCorrectionToggle(event.target.checked)}
                  />
                  <span>Corrections</span>
                </label>
                <label className="feature-switch">
                  <input
                    type="checkbox"
                    checked={isRewriteEnabled}
                    onChange={(event) => setIsRewriteEnabled(event.target.checked)}
                  />
                  <span>Rewrite</span>
                </label>
              </div>
            </div>

            <div className="editor-canvas">
              <PlainTextPlugin
                contentEditable={
                  <ContentEditable
                    className="editor-input"
                    aria-label="Realtime editor"
                    spellCheck
                  />
                }
                placeholder={null}
                ErrorBoundary={LexicalErrorBoundary}
              />
              <HistoryPlugin />
              <AutocompletePlugin
                enabled={isAutocompleteEnabled}
                onUserDismiss={handleGhostDismiss}
                tokenChunk={tokenChunk}
              />
              {isCorrectionEnabled ? (
                <CorrectionSuggestionsPlugin
                  suggestions={correctionSuggestions}
                  onAccept={handleCorrectionAccept}
                  onDismiss={handleCorrectionDismiss}
                />
              ) : null}
              {isRewriteEnabled ? (
                <FloatingToolbarPlugin
                  onRewriteRequest={handleRewriteRequest}
                  rewriteChunks={rewriteChunks}
                  rewriteDoneId={rewriteDoneId}
                />
              ) : null}
            </div>

            <div className="editor-status-grid" aria-label="Editor stats and logs">
              <div className="status-card">
                <span className="status-card-label">Server</span>
                <strong>{status}</strong>
              </div>
              <div className="status-card">
                <span className="status-card-label">Last event</span>
                <strong>{lastMessage?.type ?? 'none'}</strong>
              </div>
              <div className="status-card">
                <span className="status-card-label">Corrections</span>
                <strong>{correctionSuggestions.length}</strong>
              </div>
              <div className="activity-card">
                <div className="activity-card-header">
                  <span className="status-card-label">Activity</span>
                  <span>{activityLog.length}</span>
                </div>
                {activityLog.length > 0 ? (
                  <ol className="activity-list">
                    {activityLog.map((entry) => (
                      <li key={entry.id}>
                        <span>{entry.time}</span>
                        <strong>{entry.label}</strong>
                        <em>{entry.detail}</em>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="activity-empty">No server activity yet.</p>
                )}
              </div>
            </div>
          </div>

          <OnChangePlugin
            ignoreSelectionChange={false}
            onChange={(editorState) => {
              let shouldAutocomplete = false
              let shouldCorrect = false
              let text = ''
              let cursorCharIndex = 0

              editorState.read(() => {
                const selection = $getSelection()
                const isCollapsedRange =
                  $isRangeSelection(selection) && selection.isCollapsed()

                text = getGhostFreeTextContent($getRoot())
                cursorCharIndex = getCursorCharacterIndex()
                shouldAutocomplete = isAutocompleteEnabled && isCollapsedRange
                shouldCorrect =
                  isCorrectionEnabled && isCollapsedRange && Boolean(text.trim())
              })

              setDocumentStats(getDocumentStats(text))

              if (suppressEditorSyncRef.current) {
                if (debounceTimerRef.current !== null) {
                  window.clearTimeout(debounceTimerRef.current)
                  debounceTimerRef.current = null
                }
                if (autocompleteTimerRef.current !== null) {
                  window.clearTimeout(autocompleteTimerRef.current)
                  autocompleteTimerRef.current = null
                }
                pendingEditRef.current = null
                return
              }

              if (!shouldAutocomplete) {
                if (autocompleteTimerRef.current !== null) {
                  window.clearTimeout(autocompleteTimerRef.current)
                  autocompleteTimerRef.current = null
                }
                handleGhostDismiss()
              }

              scheduleEdit({
                text,
                cursorCharIndex,
                shouldAutocomplete,
                shouldCorrect,
              })
            }}
          />
        </section>
      </LexicalComposer>
    </main>
  )
}
