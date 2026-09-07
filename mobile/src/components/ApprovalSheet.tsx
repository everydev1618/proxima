// Tool approvals, phone-side. When Iris wants to run a shell command on the
// Mac, this sheet slides up with the actual command in mono — the terminal's
// material, judged from your pocket. First answer (phone or terminal) wins.
import * as Haptics from 'expo-haptics';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Modal, ScrollView, StyleSheet, View } from 'react-native';

import { fetchApprovals, resolveApproval, type Approval } from '../lib/api';
import { useConnection } from '../lib/connection';
import { space, useTheme } from '../lib/theme';
import { Body, Eyebrow, GhostButton, H1, Mono, PrimaryButton } from './ui';

export function ApprovalSheet() {
  const { conn } = useConnection();
  const { c } = useTheme();
  const [queue, setQueue] = useState<Approval[]>([]);
  const [busy, setBusy] = useState(false);
  const seen = useRef<Set<string>>(new Set());

  const push = useCallback((a: Approval) => {
    if (seen.current.has(a.id)) return;
    seen.current.add(a.id);
    setQueue((q) => [...q, a]);
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning).catch(() => {});
  }, []);

  const drop = useCallback((id: string) => {
    setQueue((q) => q.filter((a) => a.id !== id));
  }, []);

  useEffect(() => {
    if (!conn) return;
    let cancelled = false;
    let close: (() => void) | undefined;

    // Poll as the transport: catches approvals raised while the app was
    // backgrounded (RN pauses timers/sockets) and survives the setup→serve
    // handoff, when this endpoint doesn't exist yet.
    const poll = async () => {
      try {
        const { pending } = await fetchApprovals(conn);
        if (cancelled) return;
        for (const a of pending) push(a);
        setQueue((q) => q.filter((a) => pending.some((p) => p.id === a.id)));
      } catch {
        // Endpoint not up yet (onboarding) or Mac unreachable: nothing to do.
      }
    };
    poll();
    const t = setInterval(poll, 5000);

    // SSE makes it instant when the app is foregrounded.
    import('../lib/api').then(({ eventStream }) => {
      if (cancelled) return;
      close = eventStream(conn, '/api/v1/local/approvals/stream', (ev) => {
        if (ev.type === 'approval' && ev.approval) push(ev.approval);
        if (ev.type === 'approval_resolved' && ev.id) drop(ev.id);
      });
    });

    return () => {
      cancelled = true;
      clearInterval(t);
      close?.();
    };
  }, [conn, push, drop]);

  const current = queue[0];
  if (!conn || !current) return null;

  const answer = async (allow: boolean) => {
    setBusy(true);
    try {
      await resolveApproval(conn, current.id, allow);
    } catch {
      // Already answered on the terminal, or unreachable — either way the
      // poll/SSE will reconcile; just clear it locally.
    } finally {
      drop(current.id);
      setBusy(false);
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
    }
  };

  return (
    <Modal transparent animationType="slide" visible onRequestClose={() => answer(false)}>
      <View style={{ flex: 1, justifyContent: 'flex-end', backgroundColor: 'rgba(0,0,0,0.55)' }}>
        <View
          style={{
            backgroundColor: c.surface,
            borderTopLeftRadius: 24,
            borderTopRightRadius: 24,
            padding: space.l,
            paddingBottom: space.xl,
            gap: space.m,
            borderColor: c.hairline,
            borderWidth: StyleSheet.hairlineWidth,
          }}>
          <Eyebrow>Approval needed</Eyebrow>
          <H1 style={{ fontSize: 22, lineHeight: 28 }}>Iris wants to run {current.tool}</H1>
          <ScrollView
            style={{
              maxHeight: 180,
              backgroundColor: c.bg,
              borderRadius: 12,
              borderColor: c.hairline,
              borderWidth: StyleSheet.hairlineWidth,
            }}
            contentContainerStyle={{ padding: space.m }}>
            <Mono>{current.summary}</Mono>
          </ScrollView>
          <Body dim style={{ fontSize: 13, lineHeight: 18 }}>
            This runs on {conn.name ?? 'your computer'}. You can also answer in the terminal.
          </Body>
          <View style={{ gap: space.s }}>
            <PrimaryButton title="Allow once" onPress={() => answer(true)} busy={busy} />
            <GhostButton title="Deny" onPress={() => answer(false)} disabled={busy} />
          </View>
          {queue.length > 1 && (
            <Body dim style={{ textAlign: 'center', fontSize: 13 }}>
              {queue.length - 1} more waiting
            </Body>
          )}
        </View>
      </View>
    </Modal>
  );
}
