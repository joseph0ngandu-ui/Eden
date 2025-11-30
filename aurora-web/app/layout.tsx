import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
    title: 'Aurora Trading Dashboard',
    description: 'Real-time ML-powered trading bot monitoring',
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en">
            <body>{children}</body>
        </html>
    )
}
