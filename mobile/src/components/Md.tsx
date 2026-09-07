// Themed markdown for assistant replies: body rides the system face, code
// drops into mono on the raised surface, links glow ember.
import React from 'react';
import { StyleSheet } from 'react-native';
import Markdown from 'react-native-markdown-display';

import { fonts, useTheme } from '../lib/theme';

export function Md({ children }: { children: string }) {
  const { c } = useTheme();
  const code = {
    fontFamily: fonts.mono,
    fontSize: 13,
    backgroundColor: c.raised,
    borderColor: c.hairline,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: 8,
    padding: 10,
  };
  return (
    <Markdown
      style={{
        body: { color: c.text, fontSize: 16, lineHeight: 24 },
        paragraph: { marginTop: 0, marginBottom: 10 },
        link: { color: c.ember },
        blockquote: {
          backgroundColor: 'transparent',
          borderLeftColor: c.ember,
          borderLeftWidth: 2,
          paddingLeft: 12,
          marginLeft: 0,
          opacity: 0.9,
        },
        heading1: { fontFamily: fonts.display, color: c.text, fontSize: 22, marginBottom: 8 },
        heading2: { fontFamily: fonts.display, color: c.text, fontSize: 19, marginBottom: 6 },
        heading3: { fontFamily: fonts.display, color: c.text, fontSize: 17, marginBottom: 4 },
        code_inline: {
          fontFamily: fonts.mono,
          fontSize: 14,
          backgroundColor: c.raised,
          color: c.text,
          borderRadius: 4,
        },
        fence: code,
        code_block: code,
        bullet_list_icon: { color: c.ember },
        ordered_list_icon: { color: c.ember },
        hr: { backgroundColor: c.hairline, height: StyleSheet.hairlineWidth },
        table: { borderColor: c.hairline },
        th: { color: c.text },
        tr: { borderColor: c.hairline },
      }}>
      {children}
    </Markdown>
  );
}
