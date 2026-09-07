// Status: what's running where, and the two careful actions — clear the
// conversation, or unpair this phone.
import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import { Star } from '../components/Star';
import { Body, Card, Eyebrow, GhostButton, H1, Mono, PrimaryButton, Screen } from '../components/ui';
import { clearChat, fetchState, type LocalState } from '../lib/api';
import { baseUrl, useConnection } from '../lib/connection';
import { space, useTheme } from '../lib/theme';

export default function Status() {
  const { conn, clear } = useConnection();
  const { c } = useTheme();
  const router = useRouter();
  const [state, setState] = useState<LocalState | null>(null);
  const [reachable, setReachable] = useState(true);

  useEffect(() => {
    if (!conn) return;
    fetchState(conn)
      .then((s) => {
        setState(s);
        setReachable(true);
      })
      .catch(() => setReachable(false));
  }, [conn]);

  if (!conn) return null;

  const rows: [string, string][] = [
    ['Machine', state?.hostname ?? conn.name ?? '—'],
    ['Model', state?.model ?? '—'],
    ['Runtime', state?.server ?? '—'],
    ['Address', baseUrl(conn)],
    ['Proxima', state?.version ?? '—'],
  ];

  return (
    <Screen>
      <ScrollView contentContainerStyle={{ padding: space.l, gap: space.l }}>
        <Pressable onPress={() => router.back()} hitSlop={12}>
          <Text style={{ color: c.dim, fontSize: 16 }}>‹ Back</Text>
        </Pressable>

        <View style={{ alignItems: 'center', gap: space.m }}>
          <Star size={110} mode={reachable ? 'idle' : 'dim'} />
          <Eyebrow>{reachable ? 'Connected' : 'Unreachable'}</Eyebrow>
          <H1>{state?.hostname ?? conn.name ?? 'Your machine'}</H1>
        </View>

        <Card style={{ gap: 0, paddingVertical: 4 }}>
          {rows.map(([label, value], i) => (
            <View
              key={label}
              style={{
                flexDirection: 'row',
                justifyContent: 'space-between',
                alignItems: 'center',
                paddingVertical: 12,
                gap: space.m,
                borderTopWidth: i === 0 ? 0 : StyleSheet.hairlineWidth,
                borderTopColor: c.hairline,
              }}>
              <Body dim style={{ fontSize: 14 }}>
                {label}
              </Body>
              <Mono style={{ flexShrink: 1, textAlign: 'right' }}>{value}</Mono>
            </View>
          ))}
        </Card>

        <View style={{ gap: space.s }}>
          <GhostButton
            title="Clear the conversation"
            onPress={() =>
              Alert.alert('Clear the conversation?', 'Iris starts fresh. This can’t be undone.', [
                { text: 'Cancel', style: 'cancel' },
                {
                  text: 'Clear',
                  style: 'destructive',
                  onPress: () => clearChat(conn, 'iris').catch(() => {}),
                },
              ])
            }
          />
          <PrimaryButton
            danger
            title="Unpair this phone"
            onPress={() =>
              Alert.alert('Unpair this phone?', 'You can pair again anytime by scanning the QR.', [
                { text: 'Cancel', style: 'cancel' },
                {
                  text: 'Unpair',
                  style: 'destructive',
                  onPress: async () => {
                    await clear();
                    router.replace('/pair');
                  },
                },
              ])
            }
          />
        </View>
      </ScrollView>
    </Screen>
  );
}
