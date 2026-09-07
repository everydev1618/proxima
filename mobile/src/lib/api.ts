// Typed client for the proxima server: plain JSON calls plus the two SSE
// surfaces (chat streaming, setup/approvals events). Every request carries
// the pairing token — as a bearer header and as ?token=, since SSE
// reconnects can drop headers.
import EventSource from 'react-native-sse';

import { baseUrl, type Connection } from './connection';

// ── wire types (mirror govega/serve + cmd/proxima) ───────────

export type SetupModel = {
  id: string;
  display_name: string;
  description?: string;
  download_gb: number;
  fits: boolean;
  fit_reason?: string;
  tok_s?: number;
  starter?: boolean;
  staged?: boolean;
};

export type Progress = {
  stage?: string;
  label?: string;
  done?: number;
  total?: number;
};

export type LocalState = {
  phase: 'setup' | 'downloading' | 'starting' | 'error' | 'ready';
  hostname?: string;
  machine?: { usable_gb: number; uma: boolean };
  recommended?: string;
  models?: SetupModel[];
  progress?: Progress;
  model?: string;
  server?: string;
  version?: string;
  error?: string;
};

export type ToolActivity = {
  tool_name?: string;
  duration_ms?: number;
};

export type ChatMessage = {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  created_at?: string;
  tool_activities?: ToolActivity[];
};

export type Approval = {
  id: string;
  tool: string;
  summary: string;
  created_at?: string;
};

// ── plain JSON ───────────────────────────────────────────────

function urlFor(conn: Connection, path: string): string {
  const sep = path.includes('?') ? '&' : '?';
  return `${baseUrl(conn)}${path}${sep}token=${encodeURIComponent(conn.token)}`;
}

function authHeaders(conn: Connection): Record<string, string> {
  return { Authorization: `Bearer ${conn.token}` };
}

async function request<T>(
  conn: Connection,
  method: string,
  path: string,
  body?: unknown,
  timeoutMs = 5000,
): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(urlFor(conn, path), {
      method,
      signal: ctrl.signal,
      headers: {
        ...authHeaders(conn),
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    const text = await res.text();
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try {
        msg = JSON.parse(text).error ?? msg;
      } catch {}
      throw new Error(msg);
    }
    return (text ? JSON.parse(text) : undefined) as T;
  } finally {
    clearTimeout(timer);
  }
}

export const getJSON = <T,>(conn: Connection, path: string, timeoutMs?: number) =>
  request<T>(conn, 'GET', path, undefined, timeoutMs);
export const postJSON = <T,>(conn: Connection, path: string, body?: unknown) =>
  request<T>(conn, 'POST', path, body ?? {});
export const deleteJSON = <T,>(conn: Connection, path: string) =>
  request<T>(conn, 'DELETE', path);

export const fetchState = (conn: Connection, timeoutMs = 4000) =>
  getJSON<LocalState>(conn, '/api/v1/local/state', timeoutMs);
export const fetchHistory = (conn: Connection, agent: string) =>
  getJSON<ChatMessage[]>(conn, `/api/v1/agents/${agent}/chat`);
export const clearChat = (conn: Connection, agent: string) =>
  deleteJSON<unknown>(conn, `/api/v1/agents/${agent}/chat`);
export const startBootstrap = (conn: Connection, modelId?: string) =>
  postJSON<{ model_id: string }>(conn, '/api/v1/local/bootstrap', modelId ? { model_id: modelId } : {});
export const fetchApprovals = (conn: Connection) =>
  getJSON<{ pending: Approval[] }>(conn, '/api/v1/local/approvals');
export const resolveApproval = (conn: Connection, id: string, allow: boolean) =>
  postJSON<{ ok: boolean }>(conn, `/api/v1/local/approvals/${id}`, { allow });

// ── SSE ──────────────────────────────────────────────────────

type ChatHandlers = {
  onDelta: (text: string) => void;
  onToolStart?: (name: string) => void;
  onToolEnd?: (name: string) => void;
  onError: (message: string) => void;
  onDone: () => void;
};

type ChatEvents = 'text_delta' | 'tool_start' | 'tool_end' | 'done' | 'recalled';

// chatStream POSTs one message and streams the reply. Returns a cancel
// function. Reconnect is off: replaying a POST would re-send the message.
export function chatStream(
  conn: Connection,
  agent: string,
  message: string,
  h: ChatHandlers,
): () => void {
  const es = new EventSource<ChatEvents>(urlFor(conn, `/api/v1/agents/${agent}/chat/stream`), {
    method: 'POST',
    headers: { ...authHeaders(conn), 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
    pollingInterval: 0,
  });
  let finished = false;
  const finish = (fn: () => void) => {
    if (finished) return;
    finished = true;
    fn();
    es.removeAllEventListeners();
    es.close();
  };

  es.addEventListener('text_delta', (e) => {
    if (e.data) h.onDelta(JSON.parse(e.data).delta ?? '');
  });
  es.addEventListener('tool_start', (e) => {
    if (e.data && h.onToolStart) h.onToolStart(JSON.parse(e.data).tool_name ?? 'tool');
  });
  es.addEventListener('tool_end', (e) => {
    if (e.data && h.onToolEnd) h.onToolEnd(JSON.parse(e.data).tool_name ?? 'tool');
  });
  es.addEventListener('done', () => finish(h.onDone));
  es.addEventListener('error', (e) => {
    // Server-sent `event: error` carries data; transport errors don't.
    let msg = 'Lost the connection to your Mac.';
    if ('data' in e && e.data) {
      try {
        msg = JSON.parse(e.data as string).error ?? msg;
      } catch {}
    } else if ('message' in e && e.message) {
      msg = e.message;
    }
    finish(() => h.onError(msg));
  });

  return () => finish(() => {});
}

// eventStream opens a long-lived GET SSE (setup progress, approvals) that
// quietly reconnects. onEvent receives each parsed `data:` payload.
export function eventStream(
  conn: Connection,
  path: string,
  onEvent: (data: any) => void,
): () => void {
  const es = new EventSource(urlFor(conn, path), {
    headers: authHeaders(conn),
    pollingInterval: 4000,
  });
  es.addEventListener('message', (e) => {
    if (!e.data) return;
    try {
      onEvent(JSON.parse(e.data));
    } catch {}
  });
  es.addEventListener('error', () => {
    // Swallow: the library reconnects on pollingInterval; callers poll
    // state separately and don't need transport noise.
  });
  return () => {
    es.removeAllEventListeners();
    es.close();
  };
}
