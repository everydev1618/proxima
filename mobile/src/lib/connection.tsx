// Pairing persistence + React context. The connection (host, port, token,
// machine name) comes from the QR code proxima prints; it lives in
// SecureStore because the token authorizes shell-capable chat on the Mac.
import * as SecureStore from 'expo-secure-store';
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';

export type Connection = {
  host: string;
  port: string;
  token: string;
  name?: string;
};

const KEY = 'proxima-connection';

export function baseUrl(conn: Connection): string {
  return `http://${conn.host}:${conn.port}`;
}

// parsePairUrl decodes the QR payload: proxima://pair?h=<host>&p=<port>&t=<token>&n=<name>
export function parsePairUrl(data: string): Connection | null {
  const m = data.match(/^proxima:\/\/pair\?(.+)$/);
  if (!m) return null;
  const params: Record<string, string> = {};
  for (const kv of m[1].split('&')) {
    const eq = kv.indexOf('=');
    if (eq < 1) continue;
    params[kv.slice(0, eq)] = decodeURIComponent(kv.slice(eq + 1).replace(/\+/g, '%20'));
  }
  if (!params.h || !params.p || !params.t) return null;
  return { host: params.h, port: params.p, token: params.t, name: params.n };
}

type Ctx = {
  conn: Connection | null;
  loading: boolean;
  save: (c: Connection) => Promise<void>;
  clear: () => Promise<void>;
};

const ConnectionContext = createContext<Ctx>({
  conn: null,
  loading: true,
  save: async () => {},
  clear: async () => {},
});

export function ConnectionProvider({ children }: { children: React.ReactNode }) {
  const [conn, setConn] = useState<Connection | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const raw = await SecureStore.getItemAsync(KEY);
        if (raw) setConn(JSON.parse(raw));
      } catch {
        // Unreadable stored value: treat as unpaired.
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const save = useCallback(async (c: Connection) => {
    await SecureStore.setItemAsync(KEY, JSON.stringify(c));
    setConn(c);
  }, []);

  const clear = useCallback(async () => {
    await SecureStore.deleteItemAsync(KEY);
    setConn(null);
  }, []);

  const value = useMemo(() => ({ conn, loading, save, clear }), [conn, loading, save, clear]);
  return <ConnectionContext.Provider value={value}>{children}</ConnectionContext.Provider>;
}

export function useConnection(): Ctx {
  return useContext(ConnectionContext);
}
