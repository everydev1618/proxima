// Chat with Iris. Assistant text is set like a document (full-width, real
// markdown); your messages sit right in quiet bubbles; tool runs show as
// small mono chips. The star in the header is the status indicator — it
// tightens into a fast pulse while Iris thinks.
import * as Haptics from 'expo-haptics';
import { useRouter } from 'expo-router';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { Md } from '../components/Md';
import { Star } from '../components/Star';
import { Body, Screen } from '../components/ui';
import { chatStream, fetchHistory } from '../lib/api';
import { useConnection } from '../lib/connection';
import { fonts, space, useTheme } from '../lib/theme';

const AGENT = 'iris';

type Msg = {
  key: string;
  role: 'user' | 'assistant';
  content: string;
  tools: string[];
  streaming?: boolean;
  error?: string;
};

export default function Chat() {
  const { conn } = useConnection();
  const { c } = useTheme();
  const router = useRouter();
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const cancelRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!conn) return;
    fetchHistory(conn, AGENT)
      .then((hist) =>
        setMsgs(
          hist.map((m) => ({
            key: `h${m.id}`,
            role: m.role,
            content: m.content,
            tools: (m.tool_activities ?? []).map((t) => t.tool_name ?? 'tool'),
          })),
        ),
      )
      .catch(() => {});
    return () => cancelRef.current?.();
  }, [conn]);

  const patch = useCallback((key: string, fn: (m: Msg) => Msg) => {
    setMsgs((all) => all.map((m) => (m.key === key ? fn(m) : m)));
  }, []);

  const send = useCallback(() => {
    const text = input.trim();
    if (!text || sending || !conn) return;
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    setInput('');
    const now = Date.now();
    const asstKey = `a${now}`;
    setMsgs((all) => [
      ...all,
      { key: `u${now}`, role: 'user', content: text, tools: [] },
      { key: asstKey, role: 'assistant', content: '', tools: [], streaming: true },
    ]);
    setSending(true);
    cancelRef.current = chatStream(conn, AGENT, text, {
      onDelta: (d) => patch(asstKey, (m) => ({ ...m, content: m.content + d })),
      onToolStart: (name) => patch(asstKey, (m) => ({ ...m, tools: [...m.tools, name] })),
      onError: (message) => {
        patch(asstKey, (m) => ({ ...m, streaming: false, error: message }));
        setSending(false);
      },
      onDone: () => {
        patch(asstKey, (m) => ({ ...m, streaming: false }));
        setSending(false);
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
      },
    });
  }, [conn, input, sending, patch]);

  const inverted = useMemo(() => [...msgs].reverse(), [msgs]);

  if (!conn) return null;

  return (
    <Screen>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={{ flex: 1 }}>
        {/* Header */}
        <Pressable
          onPress={() => router.push('/status')}
          style={{
            flexDirection: 'row',
            alignItems: 'center',
            gap: space.s,
            paddingHorizontal: space.m,
            paddingVertical: space.s,
            borderBottomWidth: StyleSheet.hairlineWidth,
            borderBottomColor: c.hairline,
          }}>
          <Star size={30} mode={sending ? 'thinking' : 'idle'} />
          <View style={{ flex: 1 }}>
            <Text style={{ fontFamily: fonts.display, fontSize: 18, color: c.text }}>Proxima</Text>
            <Text style={{ fontSize: 12, color: c.faint }} numberOfLines={1}>
              {conn.name ?? `${conn.host}:${conn.port}`}
            </Text>
          </View>
          <Text style={{ color: c.faint, fontSize: 20, paddingHorizontal: space.xs }}>⋯</Text>
        </Pressable>

        {msgs.length === 0 ? (
          <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', gap: space.m, padding: space.xl }}>
            <Star size={150} mode="idle" />
            <Body dim style={{ textAlign: 'center' }}>
              Ask anything. Every word stays on {conn.name ?? 'your machine'}.
            </Body>
          </View>
        ) : (
          <FlatList
            inverted
            data={inverted}
            keyExtractor={(m) => m.key}
            contentContainerStyle={{ padding: space.m, gap: space.m }}
            renderItem={({ item }) => <Message msg={item} />}
            keyboardDismissMode="interactive"
          />
        )}

        {/* Composer */}
        <View
          style={{
            flexDirection: 'row',
            alignItems: 'flex-end',
            gap: space.s,
            padding: space.m,
            paddingTop: space.s,
          }}>
          <TextInput
            value={input}
            onChangeText={setInput}
            placeholder="Message Iris"
            placeholderTextColor={c.faint}
            multiline
            style={{
              flex: 1,
              backgroundColor: c.surface,
              borderColor: c.hairline,
              borderWidth: StyleSheet.hairlineWidth,
              borderRadius: 22,
              paddingHorizontal: 16,
              paddingTop: 12,
              paddingBottom: 12,
              maxHeight: 130,
              color: c.text,
              fontSize: 16,
            }}
          />
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Send"
            onPress={send}
            disabled={!input.trim() || sending}
            style={({ pressed }) => ({
              width: 44,
              height: 44,
              borderRadius: 22,
              backgroundColor: c.ember,
              alignItems: 'center',
              justifyContent: 'center',
              opacity: !input.trim() || sending ? 0.4 : pressed ? 0.85 : 1,
            })}>
            <Text style={{ color: c.onEmber, fontSize: 20, lineHeight: 22, fontWeight: '700' }}>↑</Text>
          </Pressable>
        </View>
      </KeyboardAvoidingView>
    </Screen>
  );
}

function Message({ msg }: { msg: Msg }) {
  const { c } = useTheme();
  if (msg.role === 'user') {
    return (
      <View style={{ alignItems: 'flex-end' }}>
        <View
          style={{
            backgroundColor: c.raised,
            borderRadius: 18,
            borderBottomRightRadius: 6,
            paddingHorizontal: 14,
            paddingVertical: 10,
            maxWidth: '82%',
          }}>
          <Text style={{ color: c.text, fontSize: 16, lineHeight: 22 }}>{msg.content}</Text>
        </View>
      </View>
    );
  }
  return (
    <View style={{ gap: 6 }}>
      {msg.tools.length > 0 && (
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 6 }}>
          {msg.tools.map((t, i) => (
            <View
              key={`${t}${i}`}
              style={{
                backgroundColor: c.emberFaint,
                borderRadius: 6,
                paddingHorizontal: 8,
                paddingVertical: 3,
              }}>
              <Text style={{ fontFamily: fonts.mono, fontSize: 11, color: c.ember }}>{t}</Text>
            </View>
          ))}
        </View>
      )}
      {msg.content ? (
        <Md>{msg.content + (msg.streaming ? ' ▍' : '')}</Md>
      ) : msg.streaming ? (
        <Text style={{ color: c.faint, fontSize: 16 }}>▍</Text>
      ) : null}
      {!!msg.error && (
        <Text style={{ color: c.danger, fontSize: 13, lineHeight: 18 }}>{msg.error}</Text>
      )}
    </View>
  );
}
