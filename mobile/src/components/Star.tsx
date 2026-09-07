// The signature element: Proxima as a living red dwarf. One orb, four
// states — dim (disconnected), idle (breathing), forming (your model is
// downloading; the star grows with progress), thinking (tight fast pulse
// while Iris streams). Everything else in the app defers to this.
import React, { useEffect } from 'react';
import Animated, {
  Easing,
  useAnimatedStyle,
  useReducedMotion,
  useSharedValue,
  withRepeat,
  withTiming,
} from 'react-native-reanimated';
import Svg, { Circle, Defs, RadialGradient, Stop } from 'react-native-svg';

import { useTheme } from '../lib/theme';

export type StarMode = 'dim' | 'idle' | 'forming' | 'thinking';

const MOTION: Record<StarMode, { duration: number; scaleTo: number; glowTo: number }> = {
  dim: { duration: 6000, scaleTo: 1.0, glowTo: 0.35 },
  idle: { duration: 3800, scaleTo: 1.05, glowTo: 1 },
  forming: { duration: 1400, scaleTo: 1.08, glowTo: 1 },
  thinking: { duration: 850, scaleTo: 1.04, glowTo: 1 },
};

export function Star({
  size = 180,
  mode = 'idle',
  progress,
}: {
  size?: number;
  mode?: StarMode;
  /** 0..1 — only used by `forming`: the star accretes as the model lands. */
  progress?: number;
}) {
  const { c } = useTheme();
  const reduced = useReducedMotion();
  const pulse = useSharedValue(1);
  const glow = useSharedValue(mode === 'dim' ? 0.35 : 0.9);

  useEffect(() => {
    const m = MOTION[mode];
    glow.value = withTiming(m.glowTo, { duration: 600 });
    if (reduced || mode === 'dim') {
      pulse.value = withTiming(1, { duration: 400 });
      return;
    }
    pulse.value = withRepeat(
      withTiming(m.scaleTo, { duration: m.duration, easing: Easing.inOut(Easing.sin) }),
      -1,
      true,
    );
  }, [mode, reduced, pulse, glow]);

  // While forming, the whole star grows from a spark to full size.
  const growth = mode === 'forming' ? 0.55 + 0.45 * Math.min(Math.max(progress ?? 0, 0), 1) : 1;

  const style = useAnimatedStyle(() => ({
    transform: [{ scale: pulse.value * growth }],
    opacity: glow.value,
  }));

  const s = size;
  return (
    <Animated.View style={[{ width: s, height: s }, style]}>
      <Svg width={s} height={s}>
        <Defs>
          <RadialGradient id="halo" cx="50%" cy="50%" r="50%">
            <Stop offset="0%" stopColor={c.emberBright} stopOpacity="0.95" />
            <Stop offset="30%" stopColor={c.ember} stopOpacity="0.55" />
            <Stop offset="65%" stopColor={c.ember} stopOpacity="0.16" />
            <Stop offset="100%" stopColor={c.ember} stopOpacity="0" />
          </RadialGradient>
          <RadialGradient id="core" cx="50%" cy="42%" r="60%">
            <Stop offset="0%" stopColor="#FFE0C2" stopOpacity="1" />
            <Stop offset="55%" stopColor={c.emberBright} stopOpacity="1" />
            <Stop offset="100%" stopColor={c.ember} stopOpacity="1" />
          </RadialGradient>
        </Defs>
        <Circle cx={s / 2} cy={s / 2} r={s / 2} fill="url(#halo)" />
        <Circle cx={s / 2} cy={s / 2} r={s * 0.17} fill="url(#core)" />
      </Svg>
    </Animated.View>
  );
}
