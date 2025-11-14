(function(){
  const el = (sel)=> document.querySelector(sel);

  function isAuthed(){
    return !!EdenStore.getToken(EDEN_CONFIG.TOKEN_KEY);
  }

  function requireAuth(){
    if(!isAuthed()){
      const dialog = document.getElementById('authDialog');
      if(dialog && !dialog.open) dialog.showModal();
      el('#userPill').textContent = 'Guest';
    } else {
      const token = EdenStore.getToken(EDEN_CONFIG.TOKEN_KEY);
      // We don't have a /me endpoint; show token presence
      el('#userPill').textContent = 'Signed in';
    }
  }

  function updateUserUI(name){ el('#userPill').textContent = name || 'Signed in'; }

  // Views
  Router.register('/dashboard', async ()=>{
    const i = await API.insights();
    return `
    <div class="grid">
      <div class="card" style="grid-column: span 4;">
        <h3>Overview</h3>
        <div class="row"><div class="badge">Trades</div><div>${i.total}</div></div>
        <div class="row"><div class="badge">Profitable</div><div>${i.done}</div></div>
        <div class="row"><div class="badge">Win Rate</div><div>${i.winRate || i.completion}%</div></div>
        <div class="row"><div class="badge">Balance</div><div>${i.accountBalance}</div></div>
        <div class="row"><div class="badge">Daily PnL</div><div>${i.dailyPnL}</div></div>
      </div>
      <div class="card" style="grid-column: span 8;">
        <h3>Open Positions</h3>
        <div id="recentTasks" class="list" style="margin-top:12px;"></div>
      </div>
    </div>`;
  });

  Router.register('/tasks', async ()=>{
    const tasks = await API.listTasks();
    const items = tasks.map(t=> `
      <div class="item">
        <input type="checkbox" ${t.done? 'checked':''} data-id="${t.id}" class="task-toggle" />
        <div style="flex:1; ${t.done? 'text-decoration: line-through; color: var(--muted)':''}">${t.title}</div>
        ${t.type==='position' ? '<button class="button secondary task-remove" data-id="'+t.id+'">Close</button>' : ''}
      </div>`).join('');

    return `<div class="card">
      <h3>Trades</h3>
      <div class="row muted">Open positions and recent history</div>
      <div class="list" style="margin-top:12px;">${items}</div>
    </div>`;
  });

  Router.register('/calendar', async ()=>{
    return `<div class="card"><h3>Calendar</h3><p class="muted">Connect your events (placeholder).</p></div>`;
  });

  Router.register('/insights', async ()=>{
    const i = await API.insights();
    return `<div class="card"><h3>Insights</h3>
      <p>Trades: <b>${i.total}</b> • Profitable: <b>${i.done}</b> • WinRate: <b>${i.winRate || i.completion}%</b></p>
      <p>Balance: <b>${i.accountBalance}</b> • DailyPnL: <b>${i.dailyPnL}</b> • TotalPnL: <b>${i.totalPnL}</b></p>
    </div>`;
  });

  Router.register('/messages', async ()=>{
    const msgs = await API.listMessages();
    const list = msgs.map(m=> `<div class="item"><div class="badge">${new Date(m.at).toLocaleTimeString()}</div><div>${m.text}</div></div>`).join('');
    return `<div class="card"><h3>Logs</h3>
      <div class="list" style="margin-top:12px;">${list || '<span class=muted>No logs</span>'}</div>
    </div>`;
  });

  Router.register('/notifications', async ()=>{
    const notifs = await API.listNotifications();
    const list = notifs.map(n=> `<div class="item"><div style="flex:1">${n.text}</div>${n.read? '<span class="badge">read</span>': '<button class="button secondary mark-read" data-id="'+n.id+'">Mark read</button>'}</div>`).join('');
    return `<div class="card"><h3>Notifications</h3><div class="list">${list || '<span class=muted>No notifications</span>'}</div></div>`;
  });

  Router.register('/settings', async ()=>{
    const s = await API.getSettings();
    return `<div class="card"><h3>Settings</h3>
      <label>Theme
        <select id="setTheme">
          <option value="dark" ${s.theme==='dark'?'selected':''}>Dark</option>
          <option value="light" ${s.theme==='light'?'selected':''}>Light</option>
        </select>
      </label>
      <div class="card" style="margin-top:12px;">
        <h3>Strategy</h3>
        <pre class="muted" style="white-space:pre-wrap;">${JSON.stringify(s.strategy || {}, null, 2)}</pre>
      </div>
      <div class="row" style="margin-top:12px;">
        <button id="saveSettings" class="button">Save</button>
      </div>
    </div>`;
  });

  Router.register('/profile', async ()=>{
    return `<div class="card"><h3>Profile</h3>
      ${isAuthed()? `<p>Status: <b>Signed in</b></p>`: '<p class="muted">You are browsing as guest.</p>'}
    </div>`;
  });

  // Global handlers after each render
  async function wireGlobal(){
    // Populate recent positions on dashboard
    const recent = document.getElementById('recentTasks');
    if(recent){
      const tasks = (await API.listTasks()).filter(t=>!t.done).slice(0,5);
      recent.innerHTML = tasks.map(t=> `<div class="item"><input type="checkbox" data-id="${t.id}" class="task-toggle" /><div style="flex:1">${t.title}</div></div>`).join('');
    }

    // Tasks view
    document.querySelectorAll('.task-toggle').forEach(cb=> cb.addEventListener('change', async (e)=>{
      await API.toggleTask(e.target.getAttribute('data-id'));
      await Router.render(); await wireGlobal();
    }));
    document.querySelectorAll('.task-remove').forEach(btn=> btn.addEventListener('click', async (e)=>{
      await API.toggleTask(e.target.getAttribute('data-id'));
      await Router.render(); await wireGlobal();
    }));

    // Notifications
    document.querySelectorAll('.mark-read').forEach(btn=> btn.addEventListener('click', async ()=>{}));

    // Settings
    const saveSettings = document.getElementById('saveSettings');
    if(saveSettings){
      saveSettings.addEventListener('click', async ()=>{
        const theme = document.getElementById('setTheme').value;
        await API.updateSettings({ theme });
        alert('Settings saved');
      });
    }

    // Auth
    const signOut = document.getElementById('signOutBtn');
    if(signOut){ signOut.onclick = async ()=>{ await API.signOut(); updateUserUI('Guest'); requireAuth(); }; }

    // Global search quick route to tasks
    const search = document.getElementById('globalSearch');
    if(search){
      search.addEventListener('change', ()=>{ location.hash = '#/tasks'; });
    }
  }

  // Auth modal wiring
  function wireAuth(){
    const dialog = document.getElementById('authDialog');
    const submit = document.getElementById('signInSubmit');
    const guest = document.getElementById('continueGuest');
    submit.addEventListener('click', async (e)=>{
      e.preventDefault();
      const email = document.getElementById('authEmail').value;
      const password = document.getElementById('authPassword').value;
      if(!email || !password) return;
      try {
        const user = await API.signIn(email, password);
        updateUserUI(user.name);
        dialog.close();
        await Router.render(); await wireGlobal();
      } catch (err) {
        alert('Login failed: ' + err.message);
      }
    });
    if(guest){ guest.addEventListener('click', ()=> dialog.close()); }
  }

  // Bootstrap
  window.addEventListener('DOMContentLoaded', async ()=>{
    wireAuth();
    requireAuth();
    await Router.render();
    await wireGlobal();
  });
})();