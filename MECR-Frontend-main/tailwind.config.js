/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#10192A',
        ink2: '#182338',
        paper: '#F6F3EC',
        paperdim: '#EAE4D3',
        line: '#D8D0BC',
        text: '#1B2333',
        textsoft: '#5B6478',
        gold: '#B8863A',
        goldsoft: '#F1E3C6',
        conflict: '#A8384A',
        conflictsoft: '#F3D9DD',
        verified: '#3E6B54',
        verifiedsoft: '#DCE9E0',
        court: '#3B5B8C',
        courtsoft: '#DCE4F3'
      },
      fontFamily: {
        serif: ['"Source Serif 4"', 'serif'],
        sans: ['Inter', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace']
      }
    }
  },
  plugins: []
}
