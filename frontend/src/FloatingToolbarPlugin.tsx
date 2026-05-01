import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  $createTextNode,
  $getSelection,
  $isRangeSelection,
  $setSelection,
  type BaseSelection,
  type LexicalEditor,
} from 'lexical'
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext'

type ToolbarPosition = {
  left: number
  top: number
}

type RewriteChunkEvent = {
  id: number
  chunk: string
}

type FloatingToolbarPluginProps = {
  onRewriteRequest: (
    highlightedText: string,
    instruction: string,
  ) => string | null
  rewriteChunk: RewriteChunkEvent | null
  rewriteDoneId: number
}

function getSelectionRect() {
  const domSelection = window.getSelection()

  if (!domSelection || domSelection.rangeCount === 0 || domSelection.isCollapsed) {
    return null
  }

  const range = domSelection.getRangeAt(0)
  const rect = range.getBoundingClientRect()

  if (rect.width === 0 && rect.height === 0) {
    return null
  }

  return rect
}

function getToolbarPosition(rect: DOMRect): ToolbarPosition {
  return {
    left: rect.left + rect.width / 2,
    top: Math.max(12, rect.top - 14),
  }
}

function clearEditorSelection(editor: LexicalEditor) {
  editor.update(() => {
    $setSelection(null)
  })
}

export function FloatingToolbarPlugin({
  onRewriteRequest,
  rewriteChunk,
  rewriteDoneId,
}: FloatingToolbarPluginProps) {
  const [editor] = useLexicalComposerContext()
  const [selectedText, setSelectedText] = useState('')
  const [instruction, setInstruction] = useState('Make this sound more professional')
  const [position, setPosition] = useState<ToolbarPosition | null>(null)
  const [storedSelection, setStoredSelection] = useState<BaseSelection | null>(null)
  const isRewritingRef = useRef(false)
  const rewriteTextRef = useRef('')

  useEffect(() => {
    return editor.registerUpdateListener(({ editorState }) => {
      editorState.read(() => {
        const selection = $getSelection()

        if (!$isRangeSelection(selection) || selection.isCollapsed()) {
          if (!isRewritingRef.current) {
            setSelectedText('')
            setPosition(null)
          }
          return
        }

        const text = selection.getTextContent()
        const rect = getSelectionRect()

        if (!text.trim() || !rect) {
          if (!isRewritingRef.current) {
            setSelectedText('')
            setPosition(null)
          }
          return
        }

        setSelectedText(text)
        setPosition(getToolbarPosition(rect))
      })
    })
  }, [editor])

  const replaceStoredSelection = useCallback(
    (text: string, selectEnd = false) => {
      editor.update(() => {
        if (!storedSelection) {
          return
        }

        $setSelection(storedSelection.clone())
        const selection = $getSelection()

        if (!$isRangeSelection(selection)) {
          return
        }

        selection.removeText()

        if (!text) {
          return
        }

        const textNode = $createTextNode(text)
        selection.insertNodes([textNode])

        if (selectEnd) {
          textNode.select(text.length, text.length)
        }
      })
    },
    [editor, storedSelection],
  )

  const handleRewrite = useCallback(() => {
    let highlightedText = ''
    let clonedSelection: BaseSelection | null = null

    editor.getEditorState().read(() => {
      const selection = $getSelection()

      if ($isRangeSelection(selection) && !selection.isCollapsed()) {
        highlightedText = selection.getTextContent()
        clonedSelection = selection.clone()
      }
    })

    if (!highlightedText.trim() || !clonedSelection) {
      return
    }

    const requestId = onRewriteRequest(highlightedText, instruction)

    if (!requestId) {
      return
    }

    setStoredSelection(clonedSelection)
    rewriteTextRef.current = ''
    isRewritingRef.current = true
    setPosition(null)
  }, [editor, instruction, onRewriteRequest])

  const handleCancel = useCallback(() => {
    setSelectedText('')
    setPosition(null)
    isRewritingRef.current = false
    rewriteTextRef.current = ''
    setStoredSelection(null)
    clearEditorSelection(editor)
  }, [editor])

  useEffect(() => {
    if (!rewriteChunk || !storedSelection) {
      return
    }

    rewriteTextRef.current += rewriteChunk.chunk
    replaceStoredSelection(rewriteTextRef.current)
  }, [replaceStoredSelection, rewriteChunk, storedSelection])

  useEffect(() => {
    if (!isRewritingRef.current || rewriteDoneId === 0) {
      return
    }

    replaceStoredSelection(rewriteTextRef.current, true)
    isRewritingRef.current = false
    rewriteTextRef.current = ''
    setStoredSelection(null)
  }, [replaceStoredSelection, rewriteDoneId])

  if (!position || !selectedText) {
    return null
  }

  return createPortal(
    <div
      className="floating-rewrite-toolbar"
      style={{
        left: position.left,
        top: position.top,
      }}
    >
      <input
        className="floating-rewrite-input"
        value={instruction}
        onChange={(event) => setInstruction(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            event.preventDefault()
            handleRewrite()
          }
          if (event.key === 'Escape') {
            event.preventDefault()
            handleCancel()
          }
        }}
      />
      <button type="button" onClick={handleRewrite}>
        Rewrite
      </button>
      <button type="button" onClick={handleCancel}>
        Cancel
      </button>
    </div>,
    document.body,
  )
}
