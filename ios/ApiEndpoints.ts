/**
 * Eden Trading Bot - API Endpoints Configuration
 * 
 * Complete endpoint definitions for iOS TypeScript application
 * Auto-generated with full URL paths for all backend endpoints
 * 
 * Last Updated: 2025-11-30
 * Backend Version: 1.0.0
 */

// ============================================================================
// BASE CONFIGURATION
// ============================================================================

/**
 * Base API URL - Update based on your network configuration:
 * - Local Development: https://localhost:8443
 * - Tailscale Network: https://desktop-p1p7892.taildbc5d3.ts.net
 * - Production: Your deployed backend URL
 */
export const API_BASE_URL = "https://desktop-p1p7892.taildbc5d3.ts.net";

/**
 * WebSocket URL for real-time notifications
 */
export const WS_BASE_URL = API_BASE_URL.replace("https://", "wss://").replace("http://", "ws://");

// ============================================================================
// ENDPOINT DEFINITIONS
// ============================================================================

/**
 * Health & System Endpoints
 */
export const HealthEndpoints = {
  /** Health check - returns {status: "ok"} */
  health: `${API_BASE_URL}/health`,
  
  /** API information */
  info: `${API_BASE_URL}/info`,
  
  /** System status (requires auth) */
  systemStatus: `${API_BASE_URL}/system/status`,
} as const;

/**
 * Authentication Endpoints
 */
export const AuthEndpoints = {
  /** Register new user - POST { email, password, full_name } */
  register: `${API_BASE_URL}/auth/register-local`,
  
  /** Login - POST { email, password } - Returns { access_token, token_type } */
  login: `${API_BASE_URL}/auth/login-local`,
} as const;

/**
 * Bot Control Endpoints
 */
export const BotEndpoints = {
  /** Get bot status (public) - GET */
  status: `${API_BASE_URL}/bot/status`,
  
  /** Start bot (requires auth) - POST */
  start: `${API_BASE_URL}/bot/start`,
  
  /** Stop bot (requires auth) - POST */
  stop: `${API_BASE_URL}/bot/stop`,
  
  /** Pause bot (requires auth) - POST */
  pause: `${API_BASE_URL}/bot/pause`,
} as const;

/**
 * Trading Endpoints
 */
export const TradeEndpoints = {
  /** Get open positions (public) - GET */
  open: `${API_BASE_URL}/trades/open`,
  
  /** Get trade history (requires auth) - GET ?limit=100 */
  history: `${API_BASE_URL}/trades/history`,
  
  /** Get recent trades (requires auth) - GET ?days=7 */
  recent: `${API_BASE_URL}/trades/recent`,
  
  /** Get trade logs (public) - GET ?limit=100 */
  logs: `${API_BASE_URL}/trades/logs`,
  
  /** Close position (requires auth) - POST { symbol } */
  close: `${API_BASE_URL}/trades/close`,
} as const;

/**
 * Order Endpoints
 */
export const OrderEndpoints = {
  /** Place test order - POST { symbol, side, volume } */
  test: `${API_BASE_URL}/orders/test`,
} as const;

/**
 * Performance Endpoints
 */
export const PerformanceEndpoints = {
  /** Get performance statistics (public) - GET */
  stats: `${API_BASE_URL}/performance/stats`,
  
  /** Get equity curve (requires auth) - GET */
  equityCurve: `${API_BASE_URL}/performance/equity-curve`,
  
  /** Get daily summary (requires auth) - GET */
  dailySummary: `${API_BASE_URL}/performance/daily-summary`,
} as const;

/**
 * Strategy Configuration Endpoints
 */
export const StrategyConfigEndpoints = {
  /** Get strategy config (public) - GET */
  getConfig: `${API_BASE_URL}/strategy/config`,
  
  /** Update strategy config (requires auth) - POST */
  updateConfig: `${API_BASE_URL}/strategy/config`,
  
  /** Get trading symbols (requires auth) - GET */
  symbols: `${API_BASE_URL}/strategy/symbols`,
} as const;

/**
 * Strategy Management Endpoints
 */
export const StrategyEndpoints = {
  /** List all strategies - GET */
  list: `${API_BASE_URL}/strategies`,
  
  /** Upload new strategy - POST */
  upload: `${API_BASE_URL}/strategies`,
  
  /** List validated strategies - GET */
  validated: `${API_BASE_URL}/strategies/validated`,
  
  /** List active strategies - GET */
  active: `${API_BASE_URL}/strategies/active`,
  
  /** Activate strategy - PUT /:id/activate */
  activate: (id: string) => `${API_BASE_URL}/strategies/${id}/activate`,
  
  /** Deactivate strategy - PUT /:id/deactivate */
  deactivate: (id: string) => `${API_BASE_URL}/strategies/${id}/deactivate`,
  
  /** Promote strategy to LIVE - PUT /:id/promote */
  promote: (id: string) => `${API_BASE_URL}/strategies/${id}/promote`,
  
  /** Update strategy policy - PATCH /:id/policy */
  updatePolicy: (id: string) => `${API_BASE_URL}/strategies/${id}/policy`,
} as const;

/**
 * MT5 Account Management Endpoints
 */
export const MT5AccountEndpoints = {
  /** List all MT5 accounts (requires auth) - GET */
  list: `${API_BASE_URL}/account/mt5`,
  
  /** Get primary MT5 account (requires auth) - GET */
  primary: `${API_BASE_URL}/account/mt5/primary`,
  
  /** Create MT5 account (requires auth) - POST */
  create: `${API_BASE_URL}/account/mt5`,
  
  /** Update MT5 account (requires auth) - PUT /:id */
  update: (id: number) => `${API_BASE_URL}/account/mt5/${id}`,
  
  /** Delete MT5 account (requires auth) - DELETE /:id */
  delete: (id: number) => `${API_BASE_URL}/account/mt5/${id}`,
} as const;

/**
 * Device & Notification Endpoints
 */
export const DeviceEndpoints = {
  /** Register device for push notifications (requires auth) - POST { token } */
  register: `${API_BASE_URL}/device/register`,
} as const;

/**
 * WebSocket Endpoints
 */
export const WebSocketEndpoints = {
  /** WebSocket notifications endpoint */
  notifications: `${WS_BASE_URL}/ws/notifications`,
  
  /** WebSocket updates (with token) */
  updates: (token: string) => `${WS_BASE_URL}/ws/updates/${token}`,
} as const;

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * Authentication Request Types
 */
export interface RegisterRequest {
  email: string;
  password: string;
  full_name: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

/**
 * MT5 Account Types
 */
export interface MT5AccountCreate {
  account_number: string;
  account_name: string;
  broker: string;
  server: string;
  password: string;
  is_primary: boolean;
}

export interface MT5Account extends Omit<MT5AccountCreate, 'password'> {
  id: number;
  user_id: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

/**
 * Trading Types
 */
export interface TestOrderRequest {
  symbol: string;
  side: 'BUY' | 'SELL';
  volume: number;
}

export interface Position {
  ticket: number;
  symbol: string;
  type: string;
  volume: number;
  open_price: number;
  current_price: number;
  profit: number;
  open_time: string;
}

export interface Trade {
  ticket: number;
  symbol: string;
  type: string;
  volume: number;
  open_price: number;
  close_price: number;
  profit: number;
  open_time: string;
  close_time: string;
}

/**
 * Performance Types
 */
export interface PerformanceStats {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_profit: number;
  average_profit: number;
  max_drawdown: number;
  sharpe_ratio: number;
}

/**
 * Bot Status Type
 */
export interface BotStatus {
  is_running: boolean;
  balance: number;
  equity: number;
  margin: number;
  free_margin: number;
  open_positions: number;
  total_profit: number;
  daily_profit: number;
  last_update: string;
}

/**
 * Strategy Config Type
 */
export interface StrategyConfig {
  strategy_name: string;
  symbols: string[];
  timeframe: string;
  parameters: Record<string, any>;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Create authenticated headers with JWT token
 */
export function createAuthHeaders(token: string): Record<string, string> {
  return {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  };
}

/**
 * Create standard headers
 */
export function createHeaders(): Record<string, string> {
  return {
    'Content-Type': 'application/json',
  };
}

/**
 * Build URL with query parameters
 */
export function buildUrlWithParams(baseUrl: string, params?: Record<string, any>): string {
  if (!params) return baseUrl;
  
  const queryString = Object.entries(params)
    .filter(([_, value]) => value !== undefined && value !== null)
    .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
    .join('&');
  
  return queryString ? `${baseUrl}?${queryString}` : baseUrl;
}

// ============================================================================
// ENDPOINT SUMMARY
// ============================================================================

/**
 * Complete list of all available endpoints
 */
export const AllEndpoints = {
  health: HealthEndpoints,
  auth: AuthEndpoints,
  bot: BotEndpoints,
  trades: TradeEndpoints,
  orders: OrderEndpoints,
  performance: PerformanceEndpoints,
  strategyConfig: StrategyConfigEndpoints,
  strategies: StrategyEndpoints,
  mt5Accounts: MT5AccountEndpoints,
  device: DeviceEndpoints,
  websocket: WebSocketEndpoints,
} as const;

/**
 * Total number of API endpoints: 40+
 * 
 * Categories:
 * - Health & System: 3 endpoints
 * - Authentication: 2 endpoints
 * - Bot Control: 4 endpoints
 * - Trading: 5 endpoints
 * - Orders: 1 endpoint
 * - Performance: 3 endpoints
 * - Strategy Config: 3 endpoints
 * - Strategy Management: 8 endpoints
 * - MT5 Accounts: 5 endpoints
 * - Device: 1 endpoint
 * - WebSocket: 2 endpoints
 */
