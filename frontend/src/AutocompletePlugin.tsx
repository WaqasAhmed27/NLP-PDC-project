import { useEffect } from 'react'
import {
  $createTextNode,
  $getRoot,
  $getSelection,
  $isElementNode,
  $isRangeSelection,
  $isTextNode,
  CLICK_COMMAND,
  COMMAND_PRIORITY_HIGH,
  KEY_DOWN_COMMAND,
  KEY_TAB_COMMAND,
  type LexicalEditor,
  type LexicalNode,
} from 'lexical'
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext'
import {
  $createAutocompleteNode,
  $isAutocompleteNode,
  type AutocompleteNode,
} from './AutocompleteNode'

export type TokenChunkEvent = {
  id: number
  chunk: string
}

type AutocompletePluginProps = {
  enabled: boolean
  onUserDismiss?: () => void
  tokenChunk: TokenChunkEvent | null
}

function isDismissKey(event: KeyboardEvent) {
  if (event.key === 'Backspace' || event.key === 'Enter') {
    return true
  }

  return event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey
}

function $findAutocompleteNode(): AutocompleteNode | null {
  const stack: LexicalNode[] = [...$getRoot().getChildren()]

  while (stack.length > 0) {
    const node = stack.shift()

    if (!node) {
      continue
    }

    if ($isAutocompleteNode(node)) {
      return node
    }

    if ($isElementNode(node)) {
      stack.unshift(...node.getChildren())
    }
  }

  return null
}

function $removeAutocompleteNode() {
  const node = $findAutocompleteNode()

  if (!node) {
    return false
  }

  node.remove()
  return true
}

function $insertOrAppendAutocompleteChunk(chunk: string) {
  const existingNode = $findAutocompleteNode()

  if (existingNode) {
    existingNode.setTextContent(existingNode.getTextContent() + chunk)
    return
  }

  const selection = $getSelection()
  const autocompleteNode = $createAutocompleteNode(chunk)

  if ($isRangeSelection(selection)) {
    selection.insertNodes([autocompleteNode])
    const previousSibling = autocompleteNode.getPreviousSibling()
    if (previousSibling && $isTextNode(previousSibling)) {
      previousSibling.selectEnd()
    }
  }
}

function $acceptAutocompleteNode() {
  const autocompleteNode = $findAutocompleteNode()

  if (!autocompleteNode) {
    return false
  }

  const acceptedText = autocompleteNode.getTextContent()
  const acceptedTextNode = $createTextNode(acceptedText)
  autocompleteNode.replace(acceptedTextNode)
  acceptedTextNode.select(acceptedText.length, acceptedText.length)
  return true
}

function registerKeyboardInterceptors(
  editor: LexicalEditor,
  onUserDismiss?: () => void,
) {
  const unregisterTab = editor.registerCommand(
    KEY_TAB_COMMAND,
    (event) => {
      const accepted = $acceptAutocompleteNode()

      if (accepted) {
        event?.preventDefault()
        onUserDismiss?.()
      }

      return accepted
    },
    COMMAND_PRIORITY_HIGH,
  )

  const unregisterDismiss = editor.registerCommand(
    KEY_DOWN_COMMAND,
    (event) => {
      if (!isDismissKey(event)) {
        return false
      }

      if ($removeAutocompleteNode()) {
        onUserDismiss?.()
      }
      return false
    },
    COMMAND_PRIORITY_HIGH,
  )

  const unregisterClickDismiss = editor.registerCommand(
    CLICK_COMMAND,
    () => {
      if ($removeAutocompleteNode()) {
        onUserDismiss?.()
      }
      return false
    },
    COMMAND_PRIORITY_HIGH,
  )

  return () => {
    unregisterTab()
    unregisterDismiss()
    unregisterClickDismiss()
  }
}

export function AutocompletePlugin({
  enabled,
  onUserDismiss,
  tokenChunk,
}: AutocompletePluginProps) {
  const [editor] = useLexicalComposerContext()

  useEffect(() => {
    if (!enabled) {
      editor.update(() => {
        $removeAutocompleteNode()
      })
      return
    }

    return registerKeyboardInterceptors(editor, onUserDismiss)
  }, [editor, enabled, onUserDismiss])

  useEffect(() => {
    if (!enabled) {
      return
    }

    const acceptTab = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') {
        return
      }

      let accepted = false

      editor.update(() => {
        accepted = $acceptAutocompleteNode()
      })

      if (accepted) {
        event.preventDefault()
        event.stopPropagation()
        event.stopImmediatePropagation()
        onUserDismiss?.()
      }
    }
    const unregisterRootListener = editor.registerRootListener(
      (rootElement, previousRootElement) => {
        previousRootElement?.removeEventListener('keydown', acceptTab, true)
        rootElement?.addEventListener('keydown', acceptTab, true)
      },
    )

    document.addEventListener('keydown', acceptTab, true)

    return () => {
      unregisterRootListener()
      editor.getRootElement()?.removeEventListener('keydown', acceptTab, true)
      document.removeEventListener('keydown', acceptTab, true)
    }
  }, [editor, enabled, onUserDismiss])

  useEffect(() => {
    if (!enabled) {
      return
    }

    const rootElement = editor.getRootElement()

    if (!rootElement) {
      return
    }

    const handlePointerDown = () => {
      editor.update(() => {
        if ($removeAutocompleteNode()) {
          onUserDismiss?.()
        }
      })
    }

    rootElement.addEventListener('pointerdown', handlePointerDown, true)

    return () => {
      rootElement.removeEventListener('pointerdown', handlePointerDown, true)
    }
  }, [editor, enabled, onUserDismiss])

  useEffect(() => {
    if (!enabled || !tokenChunk) {
      return
    }

    editor.update(() => {
      $insertOrAppendAutocompleteChunk(tokenChunk.chunk)
    })
  }, [editor, enabled, tokenChunk])

  return null
}
