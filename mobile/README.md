# Proxima mobile

The phone app for [proxima](../README.md): pair by scanning the QR the
terminal prints, and everything after that — first-run model download, chat
with Iris, tool approvals — happens on the phone.

```
cd mobile
npm install
npx expo start
```

Scan the Metro QR with Expo Go (iOS/Android) on a phone on the same network
as the Mac running `proxima`. In the app, scan the *pairing* QR from the
proxima terminal (or enter host/port/token manually — the terminal prints
those too).

## How it talks to the Mac

- `proxima` listens on `0.0.0.0:7769` by default (`-mobile=off` to disable);
  every non-loopback request needs the pairing token from
  `~/.vega/mobile-token`, sent as a bearer header or `?token=`.
- Onboarding drives `/api/v1/local/state` + `/api/v1/local/bootstrap` and
  streams download progress from `/api/v1/local/progress` (SSE).
- Chat streams from `POST /api/v1/agents/iris/chat/stream` (SSE:
  `text_delta`, `tool_start`, `tool_end`, `done`).
- Tool approvals arrive on `/api/v1/local/approvals/stream` and resolve via
  `POST /api/v1/local/approvals/{id}` — racing the terminal prompt.

## Layout

- `src/app/` — screens (expo-router): gate → pair → onboarding → chat → status
- `src/lib/` — theme tokens, pairing persistence, typed API client
- `src/components/` — `Star` (the signature orb: dim/idle/forming/thinking),
  approval sheet, themed markdown, quiet UI atoms

Checks: `npx tsc --noEmit` and `npm run lint`.
