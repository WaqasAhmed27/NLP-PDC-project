import { useCallback, useEffect, useRef, useState } from 'react'
import { $getRoot, $getSelection, $isElementNode, $isRangeSelection, $isTextNode, type ElementNode, type LexicalNode } from 'lexical'
import { LexicalComposer } from '@lexical/react/LexicalComposer'
import { PlainTextPlugin } from '@lexical/react/LexicalPlainTextPlugin'
import { ContentEditable } from '@lexical/react/LexicalContentEditable'
import { HistoryPlugin } from '@lexical/react/LexicalHistoryPlugin'
import { OnChangePlugin } from '@lexical/react/LexicalOnChangePlugin'
import { LexicalErrorBoundary } from '@lexical/react/LexicalErrorBoundary'
import { useEditorSocket, type IncomingEditorMessage } from './useEditorSocket'

type PendingEdit = {
  text: string
  cursorCharIndex: number
}

const EDIT_DEBOUNCE_MS = 50

const lexicalTheme = {
  paragraph: 'editor-paragraph',
}

function getNodeTextLength(node: LexicalNode) {
  return node.getTextContent().length
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
  const text = root.getTextContent()

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
  const pendingEditRef = useRef<PendingEdit | null>(null)
  const lastSentTextRef = useRef('')
  const [lastMessage, setLastMessage] = useState<IncomingEditorMessage | null>(null)

  const handleSocketMessage = useCallback((payload: IncomingEditorMessage) => {
    setLastMessage(payload)
    console.info('editor socket message', payload)
  }, [])

  const { status, sendMessage } = useEditorSocket({
    onMessage: handleSocketMessage,
  })

  useEffect(() => {
    return () => {
      if (debounceTimerRef.current !== null) {
        window.clearTimeout(debounceTimerRef.current)
      }
    }
  }, [])

  const flushPendingEdit = useCallback(() => {
    const pendingEdit = pendingEditRef.current
    pendingEditRef.current = null

    if (!pendingEdit) {
      return
    }

    const lastSentText = lastSentTextRef.current
    if (lastSentText === pendingEdit.text) {
      return
    }

    const editCharIndex = getFirstChangedCharacterIndex(
      lastSentText,
      pendingEdit.text,
    )

    const sent = sendMessage({
      action: 'edit',
      newText: pendingEdit.text,
      editCharIndex,
    })

    if (sent) {
      lastSentTextRef.current = pendingEdit.text
      console.info('sent editor edit', {
        new_text: pendingEdit.text,
        edit_char_index: editCharIndex,
        cursor_char_index: pendingEdit.cursorCharIndex,
      })
    }
  }, [sendMessage])

  const scheduleEdit = useCallback(
    (edit: PendingEdit) => {
      pendingEditRef.current = edit

      if (debounceTimerRef.current !== null) {
        window.clearTimeout(debounceTimerRef.current)
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
          <OnChangePlugin
            onChange={(editorState) => {
              editorState.read(() => {
                const text = $getRoot().getTextContent()
                const cursorCharIndex = getCursorCharacterIndex()
                scheduleEdit({ text, cursorCharIndex })
              })
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
