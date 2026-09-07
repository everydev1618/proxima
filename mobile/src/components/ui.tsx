// Quiet shared atoms. The star is the loud one; these stay disciplined.
import React from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
  type StyleProp,
  type TextStyle,
  type ViewStyle,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { fonts, space, useTheme } from '../lib/theme';

export function Screen({
  children,
  style,
}: {
  children: React.ReactNode;
  style?: StyleProp<ViewStyle>;
}) {
  const { c } = useTheme();
  return (
    <SafeAreaView style={[{ flex: 1, backgroundColor: c.bg }, style]}>{children}</SafeAreaView>
  );
}

export function H1({ children, style }: { children: React.ReactNode; style?: StyleProp<TextStyle> }) {
  const { c } = useTheme();
  return (
    <Text
      style={[
        { fontFamily: fonts.display, fontSize: 28, lineHeight: 34, color: c.text, letterSpacing: -0.3 },
        style,
      ]}>
      {children}
    </Text>
  );
}

export function Body({
  children,
  dim,
  style,
}: {
  children: React.ReactNode;
  dim?: boolean;
  style?: StyleProp<TextStyle>;
}) {
  const { c } = useTheme();
  return (
    <Text style={[{ fontSize: 16, lineHeight: 23, color: dim ? c.dim : c.text }, style]}>
      {children}
    </Text>
  );
}

export function Eyebrow({ children }: { children: React.ReactNode }) {
  const { c } = useTheme();
  return (
    <Text
      style={{
        fontFamily: fonts.monoMedium,
        fontSize: 11,
        letterSpacing: 1.6,
        textTransform: 'uppercase',
        color: c.ember,
      }}>
      {children}
    </Text>
  );
}

export function Mono({ children, style }: { children: React.ReactNode; style?: StyleProp<TextStyle> }) {
  const { c } = useTheme();
  return (
    <Text style={[{ fontFamily: fonts.mono, fontSize: 13, lineHeight: 19, color: c.text }, style]}>
      {children}
    </Text>
  );
}

export function Card({ children, style }: { children: React.ReactNode; style?: StyleProp<ViewStyle> }) {
  const { c } = useTheme();
  return (
    <View
      style={[
        {
          backgroundColor: c.surface,
          borderColor: c.hairline,
          borderWidth: StyleSheet.hairlineWidth,
          borderRadius: 16,
          padding: space.m,
        },
        style,
      ]}>
      {children}
    </View>
  );
}

export function PrimaryButton({
  title,
  onPress,
  busy,
  disabled,
  danger,
}: {
  title: string;
  onPress: () => void;
  busy?: boolean;
  disabled?: boolean;
  danger?: boolean;
}) {
  const { c } = useTheme();
  const off = disabled || busy;
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      disabled={off}
      style={({ pressed }) => ({
        backgroundColor: danger ? c.danger : c.ember,
        opacity: off ? 0.45 : pressed ? 0.85 : 1,
        borderRadius: 14,
        paddingVertical: 15,
        alignItems: 'center',
      })}>
      {busy ? (
        <ActivityIndicator color={c.onEmber} />
      ) : (
        <Text style={{ fontFamily: fonts.display, fontSize: 17, color: c.onEmber }}>{title}</Text>
      )}
    </Pressable>
  );
}

export function GhostButton({
  title,
  onPress,
  disabled,
}: {
  title: string;
  onPress: () => void;
  disabled?: boolean;
}) {
  const { c } = useTheme();
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => ({
        opacity: disabled ? 0.4 : pressed ? 0.6 : 1,
        borderRadius: 14,
        paddingVertical: 15,
        alignItems: 'center',
        borderWidth: StyleSheet.hairlineWidth,
        borderColor: c.hairline,
      })}>
      <Text style={{ fontFamily: fonts.display, fontSize: 17, color: c.text }}>{title}</Text>
    </Pressable>
  );
}
