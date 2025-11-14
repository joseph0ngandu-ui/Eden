(function(){
  // Token-based storage
  
  function getToken(key){
    try { return localStorage.getItem(key); }
    catch(e){ return null; }
  }
  
  function setToken(key, value){
    try { localStorage.setItem(key, value); }
    catch(e){ }
  }
  
  function removeToken(key){
    try { localStorage.removeItem(key); }
    catch(e){ }
  }
  
  window.EdenStore = {
    getToken: (key) => getToken(key),
    setToken: (key, value) => setToken(key, value),
    removeToken: (key) => removeToken(key),
    // For fallback when server is unavailable
    getLocalData(key, defaultValue = null){
      try { 
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : defaultValue;
      }
      catch(e){ return defaultValue; }
    },
    setLocalData(key, value){
      try { localStorage.setItem(key, JSON.stringify(value)); }
      catch(e){ }
    }
  };
})();
