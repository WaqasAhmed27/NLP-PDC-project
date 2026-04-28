import { useCallback, useEffect, useRef, useState } from 'react'

type EditorAction = 'edit' | 'autocomplete' | 'rewrite'
type StreamPayloadType = 'token' | 'done' | 'cancelled' | 'server_error'

type OutgoingEditorMessage = {
  action: EditorAction
  newText: string
  editCharIndex: number
}

export type IncomingEditorMessage = {
  request_id: string
  type: StreamPayloadType
  chunk: string
  latency_ms: number
}

type WireEditorPayload = {
  request_id: string
  action: EditorAction
  new_text: string
  edit_char_index: number
}

type UseEditorSocketOptions = {
  url?: string
  onMessage?: (payload: IncomingEditorMessage) => void
}

type ConnectionStatus = 'connecting' | 'open' | 'closed'

const DEFAULT_EDITOR_SOCKET_URL = 'ws://localhost:8000/ws/editor'
const RECONNECT_DELAY_MS = 500
const MAX_RECONNECT_DELAY_MS = 5000

function createRequestId() {
  if ('randomUUID' in crypto) {
    return crypto.randomUUID()
  }

  return `frontend-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function useEditorSocket({
  url = DEFAULT_EDITOR_SOCKET_URL,
  onMessage,
}: UseEditorSocketOptions = {}) {
  const socketRef = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<number | null>(null)
  const reconnectDelayRef = useRef(RECONNECT_DELAY_MS)
  const shouldReconnectRef = useRef(true)
  const onMessageRef = useRef(onMessage)
  const [status, setStatus] = useState<ConnectionStatus>('connecting')

  useEffect(() => {
    onMessageRef.current = onMessage
  }, [onMessage])

  useEffect(() => {
    shouldReconnectRef.current = true

    const clearReconnectTimer = () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
    }

    const connect = () => {
      clearReconnectTimer()
      setStatus('connecting')

      const socket = new WebSocket(url)
      socketRef.current = socket

      socket.addEventListener('open', () => {
        reconnectDelayRef.current = RECONNECT_DELAY_MS
        setStatus('open')
      })

      socket.addEventListener('message', (event) => {
        try {
          const payload = JSON.parse(event.data) as IncomingEditorMessage
          onMessageRef.current?.(payload)
        } catch (error) {
          console.error('Failed to parse editor socket message', error)
        }
      })

      socket.addEventListener('close', () => {
        if (socketRef.current === socket) {
          socketRef.current = null
        }

        setStatus('closed')

        if (!shouldReconnectRef.current) {
          return
        }

        const delay = reconnectDelayRef.current
        reconnectDelayRef.current = Math.min(delay * 2, MAX_RECONNECT_DELAY_MS)
        reconnectTimerRef.current = window.setTimeout(connect, delay)
      })

      socket.addEventListener('error', () => {
        socket.close()
      })
    }

    connect()

    return () => {
      shouldReconnectRef.current = false
      clearReconnectTimer()
      socketRef.current?.close()
      socketRef.current = null
    }
  }, [url])

  const sendMessage = useCallback((message: OutgoingEditorMessage) => {
    const socket = socketRef.current

    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return false
    }

    const payload: WireEditorPayload = {
      request_id: createRequestId(),
      action: message.action,
      new_text: message.newText,
      edit_char_index: message.editCharIndex,
    }

    socket.send(JSON.stringify(payload))
    return true
  }, [])

  return { status, sendMessage }
}
