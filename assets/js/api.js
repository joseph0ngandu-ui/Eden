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
      const data = await post('/auth/login', { email, password });
      if (data.token) {
        EdenStore.setToken(EDEN_CONFIG.TOKEN_KEY, data.token);
        if (data.refreshToken) {
          EdenStore.setToken(EDEN_CONFIG.REFRESH_TOKEN_KEY, data.refreshToken);
        }
      }
      return data.user;
    }
    
    async signOut() {
      try {
        await post('/auth/logout');
      } catch (error) {
        console.error('Error during logout:', error);
      } finally {
        EdenStore.removeToken(EDEN_CONFIG.TOKEN_KEY);
        EdenStore.removeToken(EDEN_CONFIG.REFRESH_TOKEN_KEY);
      }
    }

    // Tasks
    async listTasks() {
      return await get('/tasks');
    }
    
    async addTask(title) {
      return await post('/tasks', { title });
    }
    
    async toggleTask(id) {
      const tasks = await this.listTasks();
      const task = tasks.find(t => t.id === id);
      if (!task) throw new Error('Task not found');
      return await put(`/tasks/${id}`, { done: !task.done });
    }
    
    async removeTask(id) {
      return await del(`/tasks/${id}`);
    }

    // Notifications
    async listNotifications() {
      return await get('/notifications');
    }
    
    async markNotificationRead(id) {
      return await put(`/notifications/${id}`, { read: true });
    }

    // Settings
    async getSettings() {
      return await get('/settings');
    }
    
    async updateSettings(patch) {
      return await put('/settings', patch);
    }

    // Messages
    async listMessages() {
      return await get('/messages');
    }
    
    async sendMessage(text) {
      return await post('/messages', { text });
    }

    // Insights
    async insights() {
      return await get('/insights');
    }
  }

  window.API = new EdenAPI();
})();
