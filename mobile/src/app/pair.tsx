// Pairing: scan the QR proxima printed in the terminal, prove the connection
// with one real request, and remember it. Manual entry covers headless Macs
// and tailnet reaches where the camera can't help.
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import React, { useRef, useState } from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  TextInput,
  View,
} from 'react-native';

import { Star } from '../components/Star';
import { Body, Eyebrow, GhostButton, H1, Mono, PrimaryButton, Screen } from '../components/ui';
import { fetchState } from '../lib/api';
import { parsePairUrl, useConnection, type Connection } from '../lib/connection';
import { fonts, space, useTheme } from '../lib/theme';

export default function Pair() {
  const { c } = useTheme();
  const { save } = useConnection();
  const router = useRouter();
  const [permission, requestPermission] = useCameraPermissions();
  const [manual, setManual] = useState(false);
  const [host, setHost] = useState('');
  const [port, setPort] = useState('7769');
  const [token, setToken] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const scanning = useRef(true);

  const connect = async (candidate: Connection) => {
    setBusy(true);
    setError('');
    try {
      await fetchState(candidate);
      await save(candidate);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      router.replace('/');
    } catch (e) {
      scanning.current = true;
      setError(
        e instanceof Error && e.message.includes('401')
          ? 'That token was refused. Scan the current QR — a fresh one prints each time proxima starts.'
          : 'Found the code, but nothing answered on that address. Is proxima running, on this network?',
      );
    } finally {
      setBusy(false);
    }
  };

  const onScan = ({ data }: { data: string }) => {
    if (!scanning.current || busy) return;
    const parsed = parsePairUrl(data);
    if (!parsed) return; // not our QR; keep looking
    scanning.current = false;
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    connect(parsed);
  };

  return (
    <Screen>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={{ flex: 1 }}>
        <ScrollView
          contentContainerStyle={{ padding: space.l, gap: space.l, flexGrow: 1 }}
          keyboardShouldPersistTaps="handled">
          <View style={{ alignItems: 'center', marginTop: space.l, gap: space.m }}>
            <Star size={120} mode="idle" />
            <Eyebrow>The nearest star</Eyebrow>
            <H1 style={{ textAlign: 'center' }}>Proxima runs on your own machine</H1>
            <Body dim style={{ textAlign: 'center' }}>
              In a terminal on your computer, run <Mono>proxima</Mono>. Then scan the QR code it
              prints.
            </Body>
          </View>

          {!manual &&
            (permission?.granted ? (
              <View
                style={{
                  borderRadius: 20,
                  overflow: 'hidden',
                  aspectRatio: 1,
                  borderWidth: StyleSheet.hairlineWidth,
                  borderColor: c.hairline,
                }}>
                <CameraView
                  style={{ flex: 1 }}
                  facing="back"
                  barcodeScannerSettings={{ barcodeTypes: ['qr'] }}
                  onBarcodeScanned={onScan}
                />
              </View>
            ) : (
              <PrimaryButton
                title="Open the camera"
                onPress={() => requestPermission()}
                busy={busy}
              />
            ))}

          {manual && (
            <View style={{ gap: space.s }}>
              <Field label="Host" value={host} onChange={setHost} placeholder="192.168.1.20" />
              <Field label="Port" value={port} onChange={setPort} placeholder="7769" />
              <Field label="Token" value={token} onChange={setToken} placeholder="from the terminal" />
              <PrimaryButton
                title="Connect"
                busy={busy}
                disabled={!host || !port || !token}
                onPress={() => connect({ host: host.trim(), port: port.trim(), token: token.trim() })}
              />
            </View>
          )}

          {!!error && (
            <Body style={{ color: c.danger, textAlign: 'center', fontSize: 14, lineHeight: 20 }}>
              {error}
            </Body>
          )}

          <View style={{ marginTop: 'auto' }}>
            <GhostButton
              title={manual ? 'Scan the QR instead' : 'Enter it manually'}
              onPress={() => setManual((m) => !m)}
            />
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </Screen>
  );
}

function Field({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
}) {
  const { c } = useTheme();
  return (
    <View style={{ gap: 6 }}>
      <Body dim style={{ fontSize: 13 }}>
        {label}
      </Body>
      <TextInput
        value={value}
        onChangeText={onChange}
        placeholder={placeholder}
        placeholderTextColor={c.faint}
        autoCapitalize="none"
        autoCorrect={false}
        style={{
          backgroundColor: c.surface,
          borderColor: c.hairline,
          borderWidth: StyleSheet.hairlineWidth,
          borderRadius: 12,
          padding: 14,
          color: c.text,
          fontFamily: fonts.mono,
          fontSize: 15,
        }}
      />
    </View>
  );
}
