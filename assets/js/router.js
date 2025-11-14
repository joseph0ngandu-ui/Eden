(function(){
  const routes = {};
  const notFound = ()=> `<div class="card"><h3>Not found</h3><p class="muted">The page doesn't exist.</p></div>`;

  function register(path, render){ routes[path] = render; }
  function current(){ return location.hash.replace('#','') || '/dashboard'; }

  async function render(){
    const view = document.getElementById('view');
    const r = routes[current()] || notFound;
    view.innerHTML = await r();
  }

  window.Router = { register, render };

  window.addEventListener('hashchange', render);
  window.addEventListener('DOMContentLoaded', render);
})();
