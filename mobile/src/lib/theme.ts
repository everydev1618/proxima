// Proxima's visual system. The subject is a red dwarf star — the nearest one.
// Dark is the primary mode (deep space, warm starlight text, ember accent);
// light is warm paper with the same ember. One signature element carries the
// identity: the star orb (components/Star.tsx). Everything else stays quiet.
import { useColorScheme } from 'react-native';

export const palettes = {
  dark: {
    bg: '#0B0B10',
    surface: '#14141B',
    raised: '#1C1C25',
    hairline: '#26262F',
    text: '#ECEAE6',
    dim: '#96939C',
    faint: '#5D5B64',
    ember: '#E85D3D',
    emberBright: '#FFB27A',
    emberFaint: 'rgba(232, 93, 61, 0.14)',
    success: '#7BC98A',
    danger: '#E0564C',
    onEmber: '#160905',
  },
  light: {
    bg: '#FAF7F2',
    surface: '#FFFFFF',
    raised: '#F1EDE6',
    hairline: '#E4DFD6',
    text: '#211E1A',
    dim: '#6E6961',
    faint: '#A39D93',
    ember: '#CE4526',
    emberBright: '#E8703F',
    emberFaint: 'rgba(206, 69, 38, 0.10)',
    success: '#3E8A50',
    danger: '#C23B31',
    onEmber: '#FFF6F0',
  },
};

export type Palette = typeof palettes.dark;

export function useTheme(): { c: Palette; scheme: 'dark' | 'light' } {
  const scheme = useColorScheme() === 'light' ? 'light' : 'dark';
  return { c: palettes[scheme], scheme };
}

// Type roles. Display carries personality (headings, the big download
// number); mono is for machine material — commands, tokens, model ids.
// Body text rides the native system face for legibility and speed.
export const fonts = {
  display: 'SpaceGrotesk_500Medium',
  displayBold: 'SpaceGrotesk_700Bold',
  mono: 'IBMPlexMono_400Regular',
  monoMedium: 'IBMPlexMono_500Medium',
};

export const space = {
  xs: 4,
  s: 8,
  m: 16,
  l: 24,
  xl: 40,
  xxl: 64,
};
