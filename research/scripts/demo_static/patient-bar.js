/* Панель пациента над вьюером: чей это снимок и что ещё у него есть.
 *
 * Зачем. Открыв исследование, врач терял контекст: на экране зубы, а чьи —
 * не написано. Чтобы сравнить КЛКТ со сканом одного человека, приходилось
 * возвращаться в список пациентов и заходить заново — и так на каждое
 * сравнение.
 *
 * Один файл на оба вьюера намеренно. Панель одинаковая, а копия в каждом
 * вьюере разошлась бы: в этом проекте так уже случилось со справочником
 * ролей и с палитрой.
 *
 * Подключение:
 *     <script src="/patient-bar.js"></script>
 *     patientBar.mount(caseId, элемент);   // элемент — куда вставить
 */
(function () {
  const KIND_SHORT = { cbct: "КЛКТ", ios: "Скан", opg: "ОПТГ",
                       periapical: "Прицельный" };

  function esc(s) {
    return String(s ?? "").replace(/[&<>"]/g,
      c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  function style() {
    if (document.getElementById("patient-bar-css")) return;
    const el = document.createElement("style");
    el.id = "patient-bar-css";
    el.textContent = `
      .pbar {
        display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
        padding: 8px 14px; border-bottom: 1px solid var(--line, #242b35);
        background: var(--panel-2, #1a2028); font-size: 13px;
      }
      .pbar a.pb-back {
        color: var(--muted, #8e99a8); text-decoration: none; font-size: 12.5px;
        border: 1px solid var(--line, #242b35); border-radius: 8px;
        padding: 4px 9px;
        /* Не переносится и не сжимается: в одной строке с остальным
           «← Карточка» ломалось на два слова и тянуло шапку вверх. */
        white-space: nowrap; flex: none;
      }
      .pbar a.pb-back:hover { color: var(--text, #e6eaf0); }
      .pbar .pb-name { font-weight: 600; white-space: nowrap; flex: none; }
      .pbar .pb-sub { color: var(--muted, #8e99a8); font-size: 12px;
                      white-space: nowrap; flex: none; }
      .pbar .pb-studies { display: flex; gap: 4px; flex: none; }
      /* ЖАЛОБА ПАЦИЕНТА — то, с чего начинается приём.
         Врач открывает снимок, уже зная, на что человек жалуется, и
         смотрит в первую очередь туда. Без этой строки разбор начинался с
         абстрактного «вот 30 зубов». Курсив и приглушённый цвет: это
         слова пациента, а не наш вывод. */
      .pbar .pb-cmp {
        margin-left: auto; color: var(--muted, #8e99a8); font-style: italic;
        font-size: 12.5px; white-space: nowrap; overflow: hidden;
        text-overflow: ellipsis;
        /* Сжимается первой: имя пациента и номер карты обрезать нельзя, а
           жалобу можно — она целиком висит в подсказке. */
        flex: 0 1 auto; min-width: 0;
      }
      .pbar .pb-st {
        border: 1px solid var(--line, #242b35); border-radius: 8px;
        padding: 4px 11px; font-size: 12px; color: var(--muted, #8e99a8);
        text-decoration: none; white-space: nowrap;
      }
      .pbar .pb-st:hover { color: var(--text, #e6eaf0);
        border-color: var(--line-2, #35404d); }
      .pbar .pb-st.on {
        color: var(--accent, #2fc4b2); border-color: rgba(47,196,178,.45);
        background: rgba(47,196,178,.12);
      }
      /* Исследование, которое ещё считается или упало, кликать незачем —
         показываем состояние, а не ведём в пустой экран. */
      .pbar .pb-st.busy { opacity: .55; pointer-events: none; }
    `;
    document.head.appendChild(el);
  }

  async function mount(caseId, target) {
    if (!caseId || !target) return null;
    style();
    let data;
    try {
      const r = await fetch(`/api/cases/${encodeURIComponent(caseId)}/owner`);
      if (!r.ok) return null;
      data = await r.json();
    } catch (e) { return null; }
    if (!data || !data.patient) return null;

    const p = data.patient;
    const parts = [];
    if (p.age) parts.push(`${p.age} лет`);
    if (p.card_no) parts.push(`карта ${esc(p.card_no)}`);

    // Если у пациента несколько исследований одного типа, «Скан, Скан, Скан»
    // ничего не различает — добавляем дату. При одном таком типе дата лишняя.
    const counts = {};
    for (const s of data.studies || []) counts[s.kind] = (counts[s.kind] || 0) + 1;

    const chips = (data.studies || []).map(s => {
      const busy = s.status !== "ready";
      let label = KIND_SHORT[s.kind] || s.kind;
      if (counts[s.kind] > 1) {
        const d = new Date(s.taken_at || s.uploaded_at);
        if (!isNaN(d)) label += " " + d.toLocaleDateString("ru-RU",
          { day: "2-digit", month: "2-digit" });
      }
      const title = busy ? `${s.kind_label}: ${s.status}` : s.kind_label;
      return `<a class="pb-st${s.current ? " on" : ""}${busy ? " busy" : ""}"
                 href="${busy ? "#" : s.view_url}" title="${esc(title)}"
                >${label}${busy ? " ·" : ""}</a>`;
    }).join("");

    // Жалоба — к ЭТОМУ исследованию, не к последнему: врач может открыть
    // прошлогодний снимок, и чужая жалоба над ним была бы враньём.
    const cmp = String((data.study || {}).complaints || "").trim();

    const bar = document.createElement("div");
    bar.className = "pbar";
    bar.innerHTML = `
      <a class="pb-back" href="/patients/${p.id}">← Карточка</a>
      <span class="pb-name">${esc(p.display)}</span>
      <span class="pb-sub">${parts.join(" · ")}</span>
      <span class="pb-studies">${chips}</span>
      ${cmp ? `<span class="pb-cmp" title="Жалоба пациента">${esc(cmp)}</span>` : ""}`;
    target.prepend(bar);
    return bar;
  }

  window.patientBar = { mount };
})();
