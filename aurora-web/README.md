# Aurora Trading Dashboard

Real-time ML-powered trading bot monitoring dashboard built with Next.js 14.

## Features

- **Real-time Updates**: WebSocket connection for live bot  status
- **120fps Performance**: Optimized animations using CSS transforms and hardware acceleration
- **ML Integration**: Monitors ML Risk Manager decisions
- **Responsive Design**: Works on desktop, tablet, and mobile

## Quick Start

```bash
# Install dependencies
npm install

# Create .env.local (copy from .env.example)
cp .env.example .env.local

# Update NEXT_PUBLIC_API_URL in .env.local if needed

# Run development server
npm run dev

# Build for production
npm run build
npm start
```

## Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/joseph0ngandu-ui/Aurora-Web-App)

1. Push to GitHub repository
2. Import to Vercel
3. Add environment variable: `NEXT_PUBLIC_API_URL`
4. Deploy

## Architecture

- **Next.js 14**: App Router for optimal performance
- **TypeScript**: Full type safety
- **Tailwind CSS**: Utility-first styling with custom design system
- **WebSocket**: Real-time bot status updates
-** API Client**: Type-safe Eden backend integration

## Performance Optimizations

- CSS containment for layout isolation
- `will-change` and `translateZ(0)` for hardware acceleration
- Optimized re-renders with React memoization
- Lazy loading for non-critical components
- SWC minification in production

## Design System

- **Aurora Blue**: Primary brand color
- **Dark Mode**: Optimized for long trading sessions
- **Smooth Animations**: 120fps-capable transitions
- **Glassmorphism**: Modern UI aesthetics

---

Built with ❤️ by Joseph Ngandu
