// API Client for Eden Backend
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://localhost:8443';

export interface BotStatus {
    is_running: boolean;
    balance: number;
    equity: number;
    open_positions: number;
    daily_pnl: number;
    total_trades: number;
    win_rate: number;
}

export interface Trade {
    id: string;
    symbol: string;
    type: string;
    entry_price: number;
    exit_price: number | null;
    profit: number;
    timestamp: string;
}

export interface PerformanceStats {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    total_trades: number;
}

class EdenAPI {
    private token: string | null = null;

    async login(email: string, password: string) {
        const res = await fetch(`${API_BASE}/auth/login-local`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        });

        if (!res.ok) throw new Error('Login failed');

        const data = await res.json();
        this.token = data.access_token;
        if (typeof window !== 'undefined') {
            localStorage.setItem('token', data.access_token);
        }
        return data;
    }

    private async authFetch(endpoint: string, options: RequestInit = {}) {
        const token = this.token || (typeof window !== 'undefined' ? localStorage.getItem('token') : null);

        const res = await fetch(`${API_BASE}${endpoint}`, {
            ...options,
            headers: {
                ...options.headers,
                ...(token && { 'Authorization': `Bearer ${token}` }),
                'Content-Type': 'application/json',
            },
        });

        if (!res.ok) throw new Error(`API error: ${res.status}`);
        return res.json();
    }

    async getBotStatus(): Promise<BotStatus> {
        return this.authFetch('/bot/status');
    }

    async getTradeHistory(limit = 100): Promise<Trade[]> {
        return this.authFetch(`/trades/history?limit=${limit}`);
    }

    async getPerformanceStats(): Promise<PerformanceStats> {
        return this.authFetch('/performance/stats');
    }

    async startBot() {
        return this.authFetch('/bot/start', { method: 'POST' });
    }

    async stopBot() {
        return this.authFetch('/bot/stop', { method: 'POST' });
    }

    // WebSocket for real-time updates
    connectWebSocket(onMessage: (data: any) => void) {
        const token = this.token || (typeof window !== 'undefined' ? localStorage.getItem('token') : null);
        const ws = new WebSocket(`wss://localhost:8443/ws/updates/${token}`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage(data);
        };

        return ws;
    }
}

export const api = new EdenAPI();
