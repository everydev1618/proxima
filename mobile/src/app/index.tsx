// The gate: route by what's true — unpaired → pair; paired but the server is
// mid-setup → onboarding; ready → chat; unreachable → say so and offer a way
// back. No spinners without words.
import { Redirect, useRouter } from 'expo-router';
import React, { useCallback, useEffect, useState } from 'react';
import { View } from 'react-native';

import { Star } from '../components/Star';
import { Body, GhostButton, H1, Mono, PrimaryButton, Screen } from '../components/ui';
import { fetchState } from '../lib/api';
import { baseUrl, useConnection } from '../lib/connection';
import { space } from '../lib/theme';

export default function Gate() {
  const { conn, loading, clear } = useConnection();
  const router = useRouter();
  const [unreachable, setUnreachable] = useState(false);
  const [checking, setChecking] = useState(false);

  const check = useCallback(async () => {
    if (!conn) return;
    setChecking(true);
    setUnreachable(false);
    try {
      const state = await fetchState(conn);
      router.replace(state.phase === 'ready' ? '/chat' : '/onboarding');
    } catch {
      setUnreachable(true);
    } finally {
      setChecking(false);
    }
  }, [conn, router]);

  useEffect(() => {
    if (loading || !conn) return;
    const t = setTimeout(check, 0);
    return () => clearTimeout(t);
  }, [loading, conn, check]);

  if (!loading && !conn) return <Redirect href="/pair" />;

  return (
    <Screen>
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', padding: space.l, gap: space.m }}>
        <Star mode="dim" size={140} />
        {unreachable && conn ? (
          <View style={{ gap: space.m, alignSelf: 'stretch', alignItems: 'center' }}>
            <H1 style={{ textAlign: 'center' }}>Can’t find Proxima</H1>
            <Body dim style={{ textAlign: 'center' }}>
              Nothing answered at <Mono>{baseUrl(conn)}</Mono>. Make sure proxima is running on{' '}
              {conn.name ?? 'your computer'} and this phone is on the same network.
            </Body>
            <View style={{ alignSelf: 'stretch', gap: space.s, marginTop: space.s }}>
              <PrimaryButton title="Try again" onPress={check} busy={checking} />
              <GhostButton
                title="Pair again"
                onPress={async () => {
                  await clear();
                  router.replace('/pair');
                }}
              />
            </View>
          </View>
        ) : (
          <Body dim>Looking for your star…</Body>
        )}
      </View>
    </Screen>
  );
}
