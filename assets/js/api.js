(function(){
  // Server-connected API
  
  async function request(path, options = {}){
    const url = `${EDEN_CONFIG.API_URL}${path}`;
    const token = EdenStore.getToken(EDEN_CONFIG.TOKEN_KEY);
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    
    if (EDEN_CONFIG.DEBUG) {
      console.log(`API Request: ${options.method || 'GET'} ${url}`);
    }
    
    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });
      
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.message || response.statusText);
      }
      
      const data = await response.json();
      if (EDEN_CONFIG.DEBUG) console.log('API Response:', data);
      return data;
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
  
  async function post(path, body) {
    return request(path, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }
  
  async function get(path) {
    return request(path, { method: 'GET' });
  }
  
  async function put(path, body) {
    return request(path, {
      method: 'PUT',
      body: JSON.stringify(body),
    });
  }
  
  async function del(path) {
    return request(path, { method: 'DELETE' });
  }

  class EdenAPI {
    async signIn(email, password) {
      // Note: Your backend expects email/password as UserLogin model
      const data = await post('/auth/login', { email, password });
      if (data.access_token) {
        EdenStore.setToken(EDEN_CONFIG.TOKEN_KEY, data.access_token);
        // Your backend doesn't return user data with login, create minimal user object
        return { 
          id: email, 
          email: email, 
          name: email.split('@')[0] 
        };
      }
      throw new Error('Login failed - no token received');
    }
    
    async signOut() {
      try {
        // Your backend doesn't have explicit logout endpoint, just clear token
        await post('/auth/logout', {});
      } catch (error) {
        console.error('Error during logout:', error);
      } finally {
        EdenStore.removeToken(EDEN_CONFIG.TOKEN_KEY);
        EdenStore.removeToken(EDEN_CONFIG.REFRESH_TOKEN_KEY);
      }
    }

    // Trading Data as Tasks
    async listTasks() {
      // Use bot status as task list – filter out open positions as active tasks
      try {
        const status = await get('/bot/status');
        const trades = await get('/trades/open');
        const history = await get('/trades/history?limit=20');
        
        // Map trading data to tasks format
        const taskList = [
          ...(trades.map(pos => ({
            id: pos.ticket || pos.id,
            title: `${pos.symbol} ${pos.type} ${pos.volume} @ ${pos.price_open} (${pos.profit > 0 ? '+' : ''}${pos.profit.toFixed(2)})`,
            done: false,
            type: 'position',
            data: pos
          }))),
          ...(history.slice(0, 10).map(trade => ({
            id: trade.ticket || trade.id,
            title: `${trade.symbol} ${trade.type} ${trade.volume} @ ${trade.price || trade.open} (Closed: ${trade.profit})`,
            done: true,
            type: 'trade',
            data: trade
          })))
        ];
        return taskList;
      } catch (error) {
        console.error('Error fetching trading data:', error);
        // Fallback
        return [{ id: 'demo1', title: 'Unable to connect to backend', done: false }];
      }
    }
    
    async addTask(title) {
      // For trading tasks, this would place a trade
      // For demo, just return a mock response
      return { id: crypto.randomUUID(), title, done: false, type: 'demo' };
    }
    
    async toggleTask(id) {
      // For trading, this would close a position
      try {
        const trades = await get('/trades/open');
        const position = trades.find(t => (t.ticket || t.id) === id);
        if (position) {
          await post('/trades/close', { symbol: position.symbol });
          return true;
        }
      } catch (error) {
        console.error('Error closing position:', error);
      }
      return false;
    }
    
    async removeTask(id) {
      // Not applicable for real trades – would close position instead
      return true;
    }

    // Notifications - use recent trades as notifications
    async listNotifications() {
      try {
        const trades = await get('/trades/history?limit=10');
        const botStatus = await get('/bot/status');
        
        // Map recent trades to notifications
        const notifications = trades.map(trade => ({
          id: trade.ticket || trade.id,
          text: `Trade closed: ${trade.symbol} ${trade.type} ${trade.volume} (${trade.profit > 0 ? '+' : ''}${trade.profit})`,
          read: false,
          type: 'trade',
          data: trade
        }));
        
        // Add bot status notification if running
        if (botStatus.runtime && botStatus.runtime.bot_running) {
          notifications.push({
            id: 'bot-status',
            text: `Bot is running (${botStatus.active_positions || 0} positions)`,
            read: false,
            type: 'status',
            data: botStatus
          });
        }
        
        return notifications;
      } catch (error) {
        console.error('Error fetching notifications:', error);
        return [];
      }
    }
    
    async markNotificationRead(id) {
      // For demo purposes, just return success
      // Backend doesn't have notification marking endpoint
      return true;
    }

    // Settings - use strategy config
    async getSettings() {
      try {
        const config = await get('/strategy/config');
        const symbols = await get('/strategy/symbols');
        
        return {
          theme: 'dark',
          notifications: true,
          // Add strategy settings
          strategy: config,
          symbols: symbols.symbols || [],
        };
      } catch (error) {
        console.error('Error fetching settings:', error);
        // Fallback
        return { theme: 'dark', notifications: true };
      }
    }
    
    async updateSettings(patch) {
      try {
        if (patch.strategy) {
          await post('/strategy/config', patch.strategy);
        }
        return true;
      } catch (error) {
        console.error('Error updating settings:', error);
        return false;
      }
    }

    // Messages - use trading logs as messages
    async listMessages() {
      try {
        const logs = await get('/trades/logs?limit=20');
        return logs.map(log => ({
          id: log.id || Math.random(),
          text: `${log.timestamp || ''} ${log.type || ''} ${log.comment || ''}`,
          at: log.timestamp || Date.now(),
          type: 'log',
          data: log
        }));
      } catch (error) {
        console.error('Error fetching messages:', error);
        return [];
      }
    }
    
    async sendMessage(text) {
      // Not applicable to logs - would just return mock
      return { id: Math.random(), text, at: Date.now() };
    }

    // Insights - use performance statistics
    async insights() {
      try {
        const performance = await get('/performance/stats');
        const botStatus = await get('/bot/status');
        const trades = await get('/trades/history?limit=50');
        
        // Calculate completion metrics from trades (profitable vs unprofitable)
        const profitableTrades = trades.filter(t => t.profit > 0).length;
        const totalTrades = trades.length;
        
        return {
          total: totalTrades,
          done: profitableTrades,
          completion: totalTrades ? Math.round((profitableTrades / totalTrades) * 100) : 0,
          // Additional trading metrics
          accountBalance: botStatus.account?.balance || 0,
          dailyPnL: performance.daily_pnl || 0,
          totalPnL: performance.total_pnl || 0,
          winRate: performance.win_rate || 0,
          drawdown: performance.drawdown || 0,
          accountType: botStatus.account?.broker || 'Demo'
        };
      } catch (error) {
        console.error('Error fetching insights:', error);
        // Fallback
        return {
          total: 0,
          done: 0,
          completion: 0,
          accountBalance: 0,
          dailyPnL: 0,
          totalPnL: 0,
          winRate: 0,
          drawdown: 0,
          accountType: 'Demo'
        };
      }
    }
  }

  window.API = new EdenAPI();
})();
