/* Профиль врача: кто смотрит снимок и чьё имя уходит в заключение.
 *
 * ЧТО ЭТО. Учётных записей в проекте нет (см. auth.js), поэтому профиль
 * живёт в браузере врача, а не на сервере. Это не «пока не сделали»:
 * настоящий профиль появится вместе с сервером клиники, где у врача будет
 * учётная запись, роль и право подписи. До тех пор имя в отчёте — это то,
 * что врач ввёл сам, и интерфейс говорит об этом прямо.
 *
 * ЗАЧЕМ ВООБЩЕ. Имя врача печатается в заключении и в подписи под
 * находками. Раньше его вводили заново под каждым кейсом, в свободной
 * форме — в отчётах по одному и тому же врачу имя писалось по-разному.
 * Теперь оно задаётся один раз и подставляется везде.
 */
(function () {
  const KEY = "actina_doctor";
  const OLD = "stomadent-doctor";          // имя из прежней версии подписи

  function read() {
    let d = null;
    try { d = JSON.parse(localStorage.getItem(KEY) || "null"); } catch (e) {}
    if (!d) {
      // Перенос из прошлой версии: там хранилось одно имя строкой.
      let legacy = "";
      try { legacy = localStorage.getItem(OLD) || ""; } catch (e) {}
      d = legacy ? { name: legacy } : null;
    }
    return d || { name: "", spec: "", clinic: "" };
  }

  function save(d) {
    try {
      localStorage.setItem(KEY, JSON.stringify(d));
      if (d.name) localStorage.setItem(OLD, d.name);   // подпись читает и его
    } catch (e) { /* приватный режим — профиль просто не сохранится */ }
  }

  function name() { return read().name || ""; }

  function initials(s) {
    return String(s || "").split(/\s+/).filter(Boolean).slice(0, 2)
      .map(w => w[0]).join("").toUpperCase() || "ВР";
  }

  function style() {
    if (document.getElementById("doctor-css")) return;
    const el = document.createElement("style");
    el.id = "doctor-css";
    el.textContent = `
      .doc-btn {
        display: inline-flex; align-items: center; gap: 8px;
        background: none; border: 1px solid transparent; color: var(--text);
        border-radius: 99px; padding: 3px 11px 3px 3px; font-size: 13px;
      }
      .doc-btn:hover { border-color: var(--line); background: var(--panel-2); }
      .doc-av {
        width: 28px; height: 28px; border-radius: 50%; display: grid;
        place-items: center; font-size: 11px; font-weight: 700; color: #04211d;
        background: linear-gradient(135deg, #2fc4b2, #1b6f8f); flex: none;
      }
      .doc-modal {
        position: fixed; inset: 0; z-index: 90; display: grid;
        place-items: center; padding: 18px; background: rgba(6, 9, 13, .72);
      }
      .doc-modal[hidden] { display: none; }
      .doc-box {
        width: min(460px, 100%); background: var(--panel);
        border: 1px solid var(--line); border-radius: 14px; padding: 20px;
      }
      .doc-box h2 { margin: 0 0 4px; font-size: 19px; font-weight: 600; }
      .doc-box .hint { color: var(--muted); font-size: 12.5px; margin-bottom: 14px; }
      .doc-box label { display: block; font-size: 12.5px; color: var(--muted);
                       margin: 12px 0 4px; }
      .doc-box input {
        width: 100%; font: inherit; color: var(--text); background: var(--bg);
        border: 1px solid var(--line); border-radius: 9px; padding: 9px 11px;
      }
      .doc-box input:focus { outline: none; border-color: rgba(47,196,178,.5); }
      .doc-acts { display: flex; gap: 8px; justify-content: flex-end; margin-top: 18px; }
      .doc-note {
        margin-top: 14px; font-size: 12px; color: var(--warn);
        background: rgba(214,163,39,.09); border: 1px solid rgba(214,163,39,.25);
        border-radius: 9px; padding: 8px 10px; line-height: 1.45;
      }
    `;
    document.head.appendChild(el);
  }

  function openProfile() {
    style();
    const d = read();
    const wrap = document.createElement("div");
    wrap.className = "doc-modal";
    wrap.innerHTML = `
      <div class="doc-box">
        <h2>Профиль врача</h2>
        <div class="hint">Имя из профиля подставляется в подпись под
          находками и печатается в заключении.</div>
        <label for="dn">ФИО</label>
        <input id="dn" value="${(d.name || "").replace(/"/g, "&quot;")}"
               placeholder="Иванов Иван Иванович">
        <label for="ds">Специальность</label>
        <input id="ds" value="${(d.spec || "").replace(/"/g, "&quot;")}"
               placeholder="стоматолог-терапевт, рентгенолог">
        <label for="dc">Клиника</label>
        <input id="dc" value="${(d.clinic || "").replace(/"/g, "&quot;")}"
               placeholder="название и город">
        <div class="doc-note">Учётных записей здесь нет: профиль хранится в
          этом браузере, а подпись под находками — отметка, а не
          электронная подпись.</div>
        <div class="doc-acts">
          <button class="btn-ghost" data-act="cancel">Отмена</button>
          <button class="btn" data-act="save">Сохранить</button>
        </div>
      </div>`;
    document.body.appendChild(wrap);
    const close = () => wrap.remove();
    wrap.addEventListener("click", ev => { if (ev.target === wrap) close(); });
    wrap.querySelector('[data-act="cancel"]').onclick = close;
    wrap.querySelector('[data-act="save"]').onclick = () => {
      save({
        name: wrap.querySelector("#dn").value.trim(),
        spec: wrap.querySelector("#ds").value.trim(),
        clinic: wrap.querySelector("#dc").value.trim(),
      });
      close();
      mountWho();                 // имя в шапке меняется сразу
      document.dispatchEvent(new CustomEvent("doctorchange"));
    };
    wrap.querySelector("#dn").focus();
  }

  /* Блок «кто вошёл» в шапке. Один на все экраны: копия на каждой
     странице в этом проекте уже расходилась. */
  function mountWho(id) {
    style();
    const box = document.getElementById(id || "who");
    if (!box) return;
    const ses = window.actinaAuth && actinaAuth.read();
    if (!ses) {
      if (window.actinaAuth && actinaAuth.isGuest()) {
        box.innerHTML = '<span class="doc">Режим просмотра по ссылке</span>';
      } else {
        box.innerHTML = "";
      }
      return;
    }
    const d = read();
    const who = d.name || ses.phone || "Врач";
    box.innerHTML = "";
    const btn = document.createElement("button");
    btn.className = "doc-btn";
    btn.type = "button";
    btn.title = "Профиль врача";
    btn.innerHTML = `<span class="doc-av">${initials(d.name || "Врач")}</span>`
                  + `<span class="doc">${who.replace(/[<>&]/g, "")}</span>`;
    /* Клик по имени ведёт на СТРАНИЦУ профиля, а не открывает окно: у
       профиля появились счётчики работы, язык и выход, и окном это уже не
       живёт. Окно `openProfile` оставлено — им пользуются экраны, где
       уводить врача со снимка нельзя. */
    btn.onclick = () => { location.href = "/profile"; };
    const out = document.createElement("button");
    out.className = "out";
    out.type = "button";
    out.textContent = "Выйти";
    out.onclick = () => { actinaAuth.end(); location.href = "/"; };
    box.append(btn, out);
  }

  window.actinaDoctor = { read, save, name, openProfile, mountWho };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => mountWho());
  } else {
    mountWho();
  }
})();
