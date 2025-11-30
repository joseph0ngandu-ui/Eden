'use client'

import { useEffect, useState, useRef } from 'react'
import { api, BotStatus, Trade, PerformanceStats } from '@/lib/api'

export default function DashboardPage() {
    const [status, setStatus] = useState<BotStatus | null>(null)
    const [trades, setTrades] = useState<Trade[]>([])
    const [stats, setStats] = useState<PerformanceStats | null>(null)
    const [loading, setLoading] = useState(true)
    const wsRef = useRef<WebSocket | null>(null)

    useEffect(() => {
        // Fetch initial data
        const loadData = async () => {
            try {
                const [statusData, tradeData, statsData] = await Promise.all([
                    api.getBotStatus(),
                    api.getTradeHistory(50),
                    api.getPerformanceStats(),
                ])
                setStatus(statusData)
                setTrades(tradeData)
                setStats(statsData)
                setLoading(false)
            } catch (error) {
                console.error('Failed to load data:', error)
                setLoading(false)
            }
        }

        loadData()

        // Connect WebSocket for real-time updates
        try {
            wsRef.current = api.connectWebSocket((data) => {
                if (data.type === 'bot_status_update') {
                    setStatus(data.status)
                } else if (data.type === 'new_trade') {
                    setTrades((prev) => [data.trade, ...prev].slice(0, 50))
                }
            })
        } catch (error) {
            console.error('WebSocket connection failed:', error)
        }

        return () => {
            wsRef.current?.close()
        }
    }, [])

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-aurora-background">
                <div className="text-aurora-text text-xl animate-pulse">Loading Aurora...</div>
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-aurora-background px-6 py-8">
            <div className="max-w-7xl mx-auto space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between animate-slide-up">
                    <div>
                        <h1 className="text-4xl font-bold text-aurora-text">Aurora Dashboard</h1>
                        <p className="text-aurora-textMuted mt-1">ML-Powered Trading Bot</p>
                    </div>
                    <div className="flex gap-3">
                        <button
                            onClick={() => status?.is_running ? api.stopBot() : api.startBot()}
                            className={`px-6 py-2 rounded-lg font-medium transition-all duration-200 transform hover:scale-105 active:scale-95 ${status?.is_running
                                    ? 'bg-aurora-error text-white'
                                    : 'bg-aurora-success text-white'
                                }`}
                        >
                            {status?.is_running ? 'Stop Bot' : 'Start Bot'}
                        </button>
                    </div>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <StatCard
                        title="Balance"
                        value={`$${status?.balance.toFixed(2) || '0.00'}`}
                        change={status?.daily_pnl || 0}
                        icon="💰"
                    />
                    <StatCard
                        title="Equity"
                        value={`$${status?.equity.toFixed(2) || '0.00'}`}
                        change={(status?.equity && status?.balance) ? ((status.equity - status.balance) / status.balance * 100) : 0}
                        icon="📈"
                    />
                    <StatCard
                        title="Open Positions"
                        value={status?.open_positions || 0}
                        icon="🎯"
                    />
                    <StatCard
                        title="Win Rate"
                        value={`${(status?.win_rate || 0).toFixed(1)}%`}
                        icon="🏆"
                    />
                </div>

                {/* Performance Stats */}
                {stats && (
                    <div className="bg-aurora-surface border border-aurora-border rounded-xl p-6 animate-scale-in card">
                        <h2 className="text-2xl font-semibold text-aurora-text mb-4">Performance</h2>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                            <div>
                                <p className="text-aurora-textMuted text-sm mb-1">Total Return</p>
                                <p className={`text-2xl font-bold ${stats.total_return >= 0 ? 'text-aurora-success' : 'text-aurora-error'}`}>
                                    {stats.total_return.toFixed(2)}%
                                </p>
                            </div>
                            <div>
                                <p className="text-aurora-textMuted text-sm mb-1">Sharpe Ratio</p>
                                <p className="text-2xl font-bold text-aurora-text">{stats.sharpe_ratio.toFixed(2)}</p>
                            </div>
                            <div>
                                <p className="text-aurora-textMuted text-sm mb-1">Max Drawdown</p>
                                <p className="text-2xl font-bold text-aurora-error">{stats.max_drawdown.toFixed(2)}%</p>
                            </div>
                            <div>
                                <p className="text-aurora-textMuted text-sm mb-1">Total Trades</p>
                                <p className="text-2xl font-bold text-aurora-text">{stats.total_trades}</p>
                            </div>
                        </div>
                    </div>
                )}

                {/* Recent Trades */}
                <div className="bg-aurora-surface border border-aurora-border rounded-xl p-6 animate-fade-in card">
                    <h2 className="text-2xl font-semibold text-aurora-text mb-4">Recent Trades</h2>
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                        {trades.map((trade, idx) => (
                            <TradeRow key={trade.id || idx} trade={trade} />
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}

function StatCard({ title, value, change, icon }: { title: string; value: string | number; change?: number; icon?: string }) {
    return (
        <div className="bg-aurora-surface border border-aurora-border rounded-xl p-6 animate-scale-in hover:scale-105 transition-transform duration-200 card animate-smooth">
            <div className="flex items-start justify-between mb-2">
                <p className="text-aurora-textMuted text-sm">{title}</p>
                {icon && <span className="text-2xl">{icon}</span>}
            </div>
            <p className="text-3xl font-bold text-aurora-text mb-1">{value}</p>
            {change !== undefined && (
                <p className={`text-sm font-medium ${change >= 0 ? 'text-aurora-success' : 'text-aurora-error'}`}>
                    {change >= 0 ? '+' : ''}{change.toFixed(2)}%
                </p>
            )}
        </div>
    )
}

function TradeRow({ trade }: { trade: Trade }) {
    const isProfit = trade.profit >= 0

    return (
        <div className="flex items-center justify-between p-4 bg-aurora-background rounded-lg hover:bg-opacity-80 transition-colors duration-150 animate-smooth">
            <div className="flex items-center gap-4">
                <div className={`w-3 h-3 rounded-full ${isProfit ? 'bg-aurora-success' : 'bg-aurora-error'} animate-pulse-slow`}></div>
                <div>
                    <p className="text-aurora-text font-medium">{trade.symbol}</p>
                    <p className="text-aurora-textMuted text-sm">{trade.type}</p>
                </div>
            </div>
            <div className="text-right">
                <p className={`font-semibold ${isProfit ? 'text-aurora-success' : 'text-aurora-error'}`}>
                    ${trade.profit.toFixed(2)}
                </p>
                <p className="text-aurora-textMuted text-sm">
                    {new Date(trade.timestamp).toLocaleTimeString()}
                </p>
            </div>
        </div>
    )
}
