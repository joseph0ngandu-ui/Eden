/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./components/**/*.{js,ts,jsx,tsx,mdx}",
        "./app/**/*.{js,ts,jsx,tsx,mdx }",
    ],
    theme: {
        extend: {
            colors: {
                aurora: {
                    background: 'hsl(222, 47%, 11%)',
                    surface: 'hsl(217, 33%, 17%)',
                    border: 'hsl(217, 28%, 24%)',
                    primary: 'hsl(210, 100%, 60%)',
                    text: 'hsl(213, 31%, 91%)',
                    textMuted: 'hsl(217, 20%, 65%)',
                    success: 'hsl(134, 61%, 41%)',
                    error: 'hsl(0, 93%, 71%)',
                    warning: 'hsl(36, 100%, 57%)',
                }
            },
            animation: {
                'fade-in': 'fadeIn 0.3s ease-in-out', 'slide-up': 'slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
                'scale-in': 'scaleIn 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            },
            keyframes: {
                fadeIn: {
                    '0%': { opacity: '0' },
                    '100%': { opacity: '1' },
                },
                slideUp: {
                    '0%': { transform: 'translateY(10px)', opacity: '0' },
                    '100%': { transform: 'translateY(0)', opacity: '1' },
                },
                scaleIn: {
                    '0%': { transform: 'scale(0.95)', opacity: '0' },
                    '100%': { transform: 'scale(1)', opacity: '1' },
                },
            },
        },
    },
    plugins: [],
}
