// Phone-led first run. The Mac has proxima running but no model yet; this
// screen shows what the machine can hold, takes the one-tap consent, and the
// star literally forms as the model downloads. Handoff gaps (setup server →
// real server) are absorbed by polling — the star keeps breathing.
import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { FlatList, Modal, Pressable, StyleSheet, Text, View } from 'react-native';

import { Star } from '../components/Star';
import { Body, Card, Eyebrow, GhostButton, H1, PrimaryButton, Screen } from '../components/ui';
import {
  eventStream,
  fetchState,
  startBootstrap,
  type LocalState,
  type Progress,
  type SetupModel,
} from '../lib/api';
import { useConnection } from '../lib/connection';
import { fonts, space, useTheme } from '../lib/theme';

export default function Onboarding() {
  const { conn } = useConnection();
  const { c } = useTheme();
  const router = useRouter();
  const [state, setState] = useState<LocalState | null>(null);
  const [progress, setProgress] = useState<Progress>({});
  const [picking, setPicking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  // Once a download has been seen, a dead connection means "handoff in
  // progress", not "setup hasn't started".
  const sawProgress = useRef(false);

  const poll = useCallback(async () => {
    if (!conn) return;
    try {
      const s = await fetchState(conn);
      setState(s);
      if (s.progress?.total) {
        sawProgress.current = true;
        setProgress(s.progress);
      }
      if (s.phase === 'downloading') sawProgress.current = true;
      if (s.phase === 'ready') {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
        router.replace('/chat');
      }
    } catch {
      // Unreachable: during handoff that's expected — show "starting".
      if (sawProgress.current) {
        setState((prev) => (prev ? { ...prev, phase: 'starting' } : prev));
      }
    }
  }, [conn, router]);

  useEffect(() => {
    const kick = setTimeout(poll, 0);
    const t = setInterval(poll, 2500);
    return () => {
      clearTimeout(kick);
      clearInterval(t);
    };
  }, [poll]);

  useEffect(() => {
    if (!conn) return;
    return eventStream(conn, '/api/v1/local/progress', (ev) => {
      if (ev.type === 'progress') {
        sawProgress.current = true;
        setProgress(ev);
      }
      if (ev.type === 'phase' && ev.phase) {
        setState((prev) => (prev ? { ...prev, phase: ev.phase } : prev));
        if (ev.error) setError(ev.error);
      }
    });
  }, [conn]);

  if (!conn) return null;

  const models = state?.models ?? [];
  const recommended = models.find((m) => m.id === state?.recommended) ?? models.find((m) => m.fits);
  const fitting = models.filter((m) => m.fits);

  const consent = async (modelId?: string) => {
    setBusy(true);
    setError('');
    try {
      await startBootstrap(conn, modelId);
      sawProgress.current = true;
      setState((prev) => (prev ? { ...prev, phase: 'downloading' } : prev));
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
    } catch (e) {
      // 409 = the terminal (or another phone) already said yes. Fine.
      const msg = e instanceof Error ? e.message : '';
      if (!msg.includes('already')) setError(msg || 'Could not start the download.');
    } finally {
      setBusy(false);
      setPicking(false);
    }
  };

  const phase = state?.phase ?? 'setup';
  const pct =
    progress.total && progress.total > 0
      ? Math.min(100, Math.floor(((progress.done ?? 0) * 100) / progress.total))
      : null;
  const starMode = phase === 'downloading' ? 'forming' : phase === 'starting' ? 'thinking' : 'idle';

  return (
    <Screen>
      <View style={{ flex: 1, padding: space.l, gap: space.l }}>
        <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', gap: space.l }}>
          <Star size={200} mode={starMode} progress={pct === null ? 0.05 : pct / 100} />

          {phase === 'setup' && recommended && (
            <View style={{ gap: space.m, alignSelf: 'stretch', alignItems: 'center' }}>
              <Eyebrow>{state?.hostname ?? 'your machine'}</Eyebrow>
              <H1 style={{ textAlign: 'center' }}>Give it a star</H1>
              <Card style={{ alignSelf: 'stretch', gap: space.xs }}>
                <Text style={{ fontFamily: fonts.display, fontSize: 19, color: c.text }}>
                  {recommended.display_name}
                </Text>
                <Body dim style={{ fontSize: 14, lineHeight: 20 }}>
                  {recommended.description || 'The recommended model for this machine.'}
                </Body>
                <Body dim style={{ fontSize: 14, marginTop: 4 }}>
                  {recommended.download_gb.toFixed(1)} GB download
                  {recommended.tok_s ? ` · ~${Math.round(recommended.tok_s)} tokens/sec here` : ''}
                </Body>
              </Card>
            </View>
          )}

          {phase === 'downloading' && (
            <View style={{ alignItems: 'center', gap: space.s }}>
              <Text
                style={{
                  fontFamily: fonts.displayBold,
                  fontSize: 64,
                  color: c.text,
                  fontVariant: ['tabular-nums'],
                }}>
                {pct === null ? '…' : `${pct}%`}
              </Text>
              <Body dim>
                {progress.total
                  ? `${((progress.done ?? 0) / 1e9).toFixed(1)} of ${(progress.total / 1e9).toFixed(1)} GB`
                  : 'Preparing the download'}
              </Body>
              <Eyebrow>Your star is forming</Eyebrow>
            </View>
          )}

          {phase === 'starting' && (
            <View style={{ alignItems: 'center', gap: space.s }}>
              <H1>Igniting</H1>
              <Body dim style={{ textAlign: 'center' }}>
                Loading the model into memory. The first light takes a minute.
              </Body>
            </View>
          )}

          {phase === 'error' && (
            <View style={{ alignItems: 'center', gap: space.s }}>
              <H1>That didn’t land</H1>
              <Body dim style={{ textAlign: 'center' }}>
                {error || state?.error || 'The download failed. Check the terminal on your computer.'}
              </Body>
            </View>
          )}
        </View>

        {phase === 'setup' && recommended && (
          <View style={{ gap: space.s }}>
            <PrimaryButton
              title={`Download ${recommended.display_name}`}
              onPress={() => consent(recommended.id)}
              busy={busy}
            />
            {fitting.length > 1 && (
              <GhostButton title="Choose a different model" onPress={() => setPicking(true)} />
            )}
          </View>
        )}
        {phase === 'error' && <PrimaryButton title="Try again" onPress={() => consent()} busy={busy} />}
      </View>

      <Modal visible={picking} animationType="slide" transparent onRequestClose={() => setPicking(false)}>
        <View style={{ flex: 1, justifyContent: 'flex-end', backgroundColor: 'rgba(0,0,0,0.55)' }}>
          <View
            style={{
              backgroundColor: c.surface,
              borderTopLeftRadius: 24,
              borderTopRightRadius: 24,
              padding: space.l,
              paddingBottom: space.xl,
              gap: space.m,
              maxHeight: '75%',
            }}>
            <Eyebrow>Fits {state?.hostname ?? 'this machine'}</Eyebrow>
            <FlatList
              data={fitting}
              keyExtractor={(m) => m.id}
              ItemSeparatorComponent={() => (
                <View style={{ height: StyleSheet.hairlineWidth, backgroundColor: c.hairline }} />
              )}
              renderItem={({ item }) => (
                <ModelRow model={item} onPress={() => consent(item.id)} disabled={busy} />
              )}
            />
            <GhostButton title="Cancel" onPress={() => setPicking(false)} />
          </View>
        </View>
      </Modal>
    </Screen>
  );
}

function ModelRow({
  model,
  onPress,
  disabled,
}: {
  model: SetupModel;
  onPress: () => void;
  disabled?: boolean;
}) {
  const { c } = useTheme();
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => ({ paddingVertical: space.m, opacity: pressed ? 0.6 : 1 })}>
      <Text style={{ fontFamily: fonts.display, fontSize: 17, color: c.text }}>
        {model.display_name}
        {model.starter ? '  ·  starter' : ''}
      </Text>
      <Body dim style={{ fontSize: 14, marginTop: 2 }}>
        {model.download_gb.toFixed(1)} GB
        {model.tok_s ? ` · ~${Math.round(model.tok_s)} tok/s` : ''}
        {model.fit_reason ? ` · ${model.fit_reason}` : ''}
      </Body>
    </Pressable>
  );
}
