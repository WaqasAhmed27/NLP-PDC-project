import { useCallback, useEffect, useRef, useState } from 'react'

type StreamPayloadType = 'token' | 'corrections' | 'done' | 'cancelled' | 'server_error'

type OutgoingEditorMessage = {
  action: 'edit' | 'autocomplete' | 'correct'
  newText: string
  editCharIndex: number
}

type OutgoingRewriteMessage = {
  highlightedText: string
  instruction: string
}

export type IncomingEditorMessage = {
  request_id: string
  type: StreamPayloadType
  chunk: string
  latency_ms: number
}

type WireEditorPayload = {
  request_id: string
  action: 'edit' | 'autocomplete' | 'correct'
  new_text: string
  edit_char_index: number
}

type WireRewritePayload = {
  request_id: string
  action: 'rewrite'
  text: string
  prompt: string
}

type UseEditorSocketOptions = {
  url?: string
  onMessage?: (payload: IncomingEditorMessage) => void
}

type ConnectionStatus = 'connecting' | 'open' | 'closed'

const getWebSocketUrl = () => {
  const configuredUrl = import.meta.env.VITE_EDITOR_WS_URL?.trim()

  if (configuredUrl) {
    return configuredUrl
      .replace(/^https:\/\//, 'wss://')
      .replace(/^http:\/\//, 'ws://')
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws/editor`
}

const DEFAULT_EDITOR_SOCKET_URL = getWebSocketUrl()
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

      console.info('connecting editor socket', url)
      const socket = new WebSocket(url)
      socketRef.current = socket

      socket.addEventListener('open', () => {
        console.info('editor socket open', url)
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
        console.info('editor socket closed', url)
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
        console.error('editor socket error', url)
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
      return null
    }

    const payload: WireEditorPayload = {
      request_id: createRequestId(),
      action: message.action,
      new_text: message.newText,
      edit_char_index: message.editCharIndex,
    }

    socket.send(JSON.stringify(payload))
    console.info('editor socket sent', payload)
    return payload.request_id
  }, [])

  const sendRewriteRequest = useCallback((message: OutgoingRewriteMessage) => {
    const socket = socketRef.current

    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return null
    }

    const requestId = createRequestId()
    const payload: WireRewritePayload = {
      request_id: requestId,
      action: 'rewrite',
      text: message.highlightedText,
      prompt: message.instruction,
    }

    socket.send(JSON.stringify(payload))
    console.info('editor socket sent rewrite', payload)
    return requestId
  }, [])

  return { status, sendMessage, sendRewriteRequest }
}
