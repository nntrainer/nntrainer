/* Navigation, local search and code copying. No network dependencies. */
(() => {
  const ui = JSON.parse(document.querySelector('#guide-i18n').textContent);
  // Shared IDs let language changes retain the current subsection as well.
  const syncLanguageLinks = () => {
    document.querySelectorAll('[data-language]').forEach(link => {
      const target = new URL(link.getAttribute('href'), location.href);
      target.hash = location.hash;
      link.href = target.href;
    });
  };
  syncLanguageLinks();
  window.addEventListener('hashchange', syncLanguageLinks);
  const sidebar = document.querySelector('.sidebar');
  const toggle = document.querySelector('.menu-toggle');
  const backdrop = document.querySelector('.sidebar-backdrop');
  const menu = (open) => {
    sidebar.classList.toggle('is-open', open);
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? ui.close_menu : ui.open_menu);
    backdrop.hidden = !open;
    if (open) sidebar.querySelector('a').focus();
  };
  toggle.addEventListener('click', () => menu(!sidebar.classList.contains('is-open')));
  backdrop.addEventListener('click', () => { menu(false); toggle.focus(); });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && sidebar.classList.contains('is-open')) {
      menu(false); toggle.focus();
    }
  });

  const theme = document.querySelector('.theme-toggle');
  const updateThemeLabel = () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    theme.setAttribute('aria-label', ui[`theme_${next}`]);
    theme.title = ui[`theme_${next}`];
  };
  updateThemeLabel();
  theme.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('nntr-guide-theme', next); } catch {}
    updateThemeLabel();
  });

  document.querySelectorAll('article h1[id], article h2[id]').forEach((heading) => {
    const link = document.createElement('a');
    link.className = 'heading-anchor';
    link.href = `#${heading.id}`;
    link.setAttribute('aria-label', ui.heading_link.replace('{heading}', heading.textContent));
    link.textContent = '¶';
    heading.append(link);
  });

  document.querySelectorAll('article pre').forEach((pre) => {
    const code = pre.querySelector('code');
    if (!code || code.classList.contains('language-text')) return;
    const label = document.createElement('span');
    label.className = 'code-language';
    label.textContent = code.className.replace('language-', '') || 'code';
    const button = document.createElement('button');
    button.className = 'copy-code';
    button.textContent = ui.copy;
    button.setAttribute('aria-label', ui.copy_label);
    pre.classList.add('has-copy');
    pre.append(label, button);
    button.addEventListener('click', async () => {
      const text = code.textContent;
      let copied = false;
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(text);
          copied = true;
        }
      } catch {}
      if (!copied) {
        // The LAN HTTP URL is not a secure context, so use a selection fallback.
        const area = document.createElement('textarea');
        area.value = text;
        area.style.cssText = 'position:fixed;left:-9999px;top:0';
        document.body.append(area);
        area.select();
        try { copied = document.execCommand('copy'); } catch {}
        area.remove();
        button.focus();
      }
      button.textContent = copied ? ui.copied : ui.copy_select;
      document.querySelector('#copy-status').textContent = copied ? ui.copy_success : ui.copy_failure;
      if (!copied) {
        const range = document.createRange(); range.selectNodeContents(code);
        const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      }
      setTimeout(() => { button.textContent = ui.copy; }, 2000);
    });
  });

  const dialog = document.querySelector('#search-dialog');
  const input = document.querySelector('#search-input');
  const results = document.querySelector('#search-results');
  const status = document.querySelector('#search-status');
  const openSearch = () => { dialog.showModal(); input.focus(); };
  document.querySelector('.search-open').addEventListener('click', openSearch);
  document.querySelector('.search-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (event) => { if (event.target === dialog) {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  }});
  document.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault(); if (!dialog.open) openSearch();
    }
  });
  input.addEventListener('input', () => {
    const query = input.value.trim().toLowerCase();
    results.replaceChildren();
    if (!query) { status.textContent = ui.search_empty; return; }
    const words = query.split(/\s+/);
    const matches = (window.GUIDE_SEARCH || []).map((entry) => {
      const all = `${entry.title} ${entry.heading} ${entry.text}`.toLowerCase();
      return {entry, score: words.every(word => all.includes(word))
        ? 1 + words.filter(word => `${entry.title} ${entry.heading}`.toLowerCase().includes(word)).length * 10 : 0};
    }).filter(hit => hit.score).sort((a, b) => b.score - a.score);
    status.textContent = matches.length ? ui.search_count.replace('{count}', matches.length) : ui.search_none;
    matches.slice(0, 20).forEach(({entry}) => {
      const li = document.createElement('li');
      const a = document.createElement('a'); a.href = entry.url;
      const category = document.createElement('small'); category.textContent = entry.title;
      const title = document.createElement('strong'); title.textContent = entry.heading;
      const snippet = document.createElement('p');
      const start = Math.max(0, entry.text.toLowerCase().indexOf(words[0]) - 45);
      snippet.textContent = `${start ? '…' : ''}${entry.text.slice(start, start + 180)}…`;
      a.append(category, title, snippet);
      a.addEventListener('click', () => dialog.close());
      li.append(a); results.append(li);
    });
  });
  input.addEventListener('keydown', event => {
    if (event.isComposing) return;
    if (event.key === 'ArrowDown') { event.preventDefault(); results.querySelector('a')?.focus(); }
    if (event.key === 'Enter') results.querySelector('a')?.click();
  });
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting);
      if (!visible.length) return;
      const id = visible[0].target.id;
      document.querySelectorAll('.page-toc a').forEach(a => a.classList.toggle('active', a.hash === `#${id}`));
    }, {rootMargin: '-5% 0px -65% 0px'});
    document.querySelectorAll('article h2[id]').forEach(h => observer.observe(h));
  }
})();
