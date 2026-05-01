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
import { FloatingToolbarPlugin } from './FloatingToolbarPlugin'

type PendingEdit = {
  text: string
  cursorCharIndex: number
  shouldAutocomplete: boolean
}

const EDIT_DEBOUNCE_MS = 50
const AUTOCOMPLETE_DEBOUNCE_MS = 250

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

export function Editor() {
  const debounceTimerRef = useRef<number | null>(null)
  const autocompleteTimerRef = useRef<number | null>(null)
  const pendingEditRef = useRef<PendingEdit | null>(null)
  const lastSentTextRef = useRef('')
  const lastSentCursorRef = useRef(0)
  const suppressIncomingTokensRef = useRef(false)
  const suppressEditorSyncRef = useRef(false)
  const activeRewriteRequestIdRef = useRef<string | null>(null)
  const tokenChunkIdRef = useRef(0)
  const rewriteChunkIdRef = useRef(0)
  const rewriteDoneIdRef = useRef(0)
  const [lastMessage, setLastMessage] = useState<IncomingEditorMessage | null>(null)
  const [tokenChunk, setTokenChunk] = useState<TokenChunkEvent | null>(null)
  const [rewriteChunk, setRewriteChunk] = useState<TokenChunkEvent | null>(null)
  const [rewriteDoneId, setRewriteDoneId] = useState(0)
  const [isAutocompleteEnabled, setIsAutocompleteEnabled] = useState(true)
  const [isRewriteEnabled, setIsRewriteEnabled] = useState(true)

  const handleSocketMessage = useCallback((payload: IncomingEditorMessage) => {
    setLastMessage(payload)

    if (payload.request_id === activeRewriteRequestIdRef.current) {
      if (payload.type === 'token' && payload.chunk) {
        rewriteChunkIdRef.current += 1
        setRewriteChunk({
          id: rewriteChunkIdRef.current,
          chunk: payload.chunk,
        })
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

    if (payload.type === 'token' && payload.chunk && !suppressIncomingTokensRef.current) {
      tokenChunkIdRef.current += 1
      setTokenChunk({
        id: tokenChunkIdRef.current,
        chunk: payload.chunk,
      })
    }

    console.info('editor socket message', payload)
  }, [])

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
        setTokenChunk(null)
        setRewriteChunk(null)
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
    }
  }, [scheduleAutocomplete, sendMessage])

  const scheduleEdit = useCallback(
    (edit: PendingEdit) => {
      pendingEditRef.current = edit

      if (debounceTimerRef.current !== null) {
        window.clearTimeout(debounceTimerRef.current)
      }

      if (autocompleteTimerRef.current !== null) {
        window.clearTimeout(autocompleteTimerRef.current)
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
      <header className="editor-header">
        <div>
          <p className="eyebrow">Phase 5</p>
          <h1>Realtime Editor</h1>
        </div>
        <span className={`socket-pill socket-pill-${status}`}>{status}</span>
      </header>

      <div className="feature-toggles" aria-label="Editor features">
        <label className="feature-toggle">
          <input
            type="checkbox"
            checked={isAutocompleteEnabled}
            onChange={(event) => handleAutocompleteToggle(event.target.checked)}
          />
          <span>Autocomplete</span>
        </label>
        <label className="feature-toggle">
          <input
            type="checkbox"
            checked={isRewriteEnabled}
            onChange={(event) => setIsRewriteEnabled(event.target.checked)}
          />
          <span>Rewrite toolbar</span>
        </label>
      </div>

      <LexicalComposer initialConfig={initialConfig}>
        <section className="editor-frame">
          <PlainTextPlugin
            contentEditable={
              <ContentEditable
                className="editor-input"
                aria-label="Realtime editor"
                spellCheck
              />
            }
            placeholder={
              <div className="editor-placeholder">Start typing to sync with FastAPI...</div>
            }
            ErrorBoundary={LexicalErrorBoundary}
          />
          <HistoryPlugin />
          <AutocompletePlugin
            enabled={isAutocompleteEnabled}
            onUserDismiss={handleGhostDismiss}
            tokenChunk={tokenChunk}
          />
          {isRewriteEnabled ? (
            <FloatingToolbarPlugin
              onRewriteRequest={handleRewriteRequest}
              rewriteChunk={rewriteChunk}
              rewriteDoneId={rewriteDoneId}
            />
          ) : null}
          <OnChangePlugin
            ignoreSelectionChange={false}
            onChange={(editorState) => {
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

              let shouldAutocomplete = false
              let text = ''
              let cursorCharIndex = 0

              editorState.read(() => {
                const selection = $getSelection()
                const isCollapsedRange =
                  $isRangeSelection(selection) && selection.isCollapsed()

                text = getGhostFreeTextContent($getRoot())
                cursorCharIndex = getCursorCharacterIndex()
                shouldAutocomplete = isAutocompleteEnabled && isCollapsedRange
              })

              if (!shouldAutocomplete) {
                if (autocompleteTimerRef.current !== null) {
                  window.clearTimeout(autocompleteTimerRef.current)
                  autocompleteTimerRef.current = null
                }
                handleGhostDismiss()
              }

              scheduleEdit({ text, cursorCharIndex, shouldAutocomplete })
            }}
          />
        </section>
      </LexicalComposer>

      <footer className="editor-footer">
        <span>Last server event: {lastMessage?.type ?? 'none'}</span>
        <span>{lastMessage?.chunk ?? ''}</span>
      </footer>
    </main>
  )
}
