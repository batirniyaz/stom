/* Язык интерфейса: русский (по умолчанию) и каракалпакский.
 *
 * Что переводится и что НЕТ. Переводится ОБВЯЗКА: кнопки, заголовки панелей,
 * названия вкладок, легенды, подписи управления. Клинический текст —
 * формулировки находок, план лечения, названия патологий и оговорки — остаётся
 * РУССКИМ, и это решение, а не недоделка:
 *
 *   * находки приходят с сервера уже собранными строками с числами и порогами
 *     («расстояние корень -> канал 1.2 мм, контакт <=0.5»), их перевод — это
 *     перевод медицинского заключения, а не интерфейса;
 *   * термин, переведённый неточно, в заключении опаснее непереведённого:
 *     врач читает диагноз, а не подпись кнопки;
 *   * документация клиники и план лечения ведутся на русском.
 *
 * Каракалпакский набран латиницей. ПЕРЕВОД ТРЕБУЕТ ВЫЧИТКИ НОСИТЕЛЕМ —
 * анатомические термины взяты по аналогии с казахским и узбекским, и часть из
 * них в клинике может звучать иначе. Заменять прямо в этом файле.
 */
(function () {
  const DICT = {
    kk: {
      // --- общее ---
      "Загрузить": "Júklew",
      "Примеры": "Mısallar",
      "Модель": "Model",
      "Сбросить вид": "Kóriniti qayta tiklew",
      "Кейс": "Jaǵday",
      "Пациент": "Nauqas",
      "Пациенты": "Nauqaslar",
      "Пациенты и исследования": "Nauqaslar hám tekseriwler",
      "Карточка пациента": "Nauqas kartochkası",
      "Журнал": "Jurnal",
      "Поиск по фамилии или номеру карты": "Familiya yamasa karta nomeri boyınsha izlew",
      "Новый пациент": "Jańa nauqas",
      "Фамилия": "Familiyası",
      "Имя": "Atı",
      "Год рождения": "Tuwılǵan jılı",
      "Номер карты": "Karta nomeri",
      "Заметка": "Belgi",
      "Отмена": "Biykar etiw",
      "Завести": "Qosıw",
      "Сохраняем…": "Saqlanbaqta…",
      "Все пациенты": "Barlıq nauqaslar",
      "Загрузить исследование": "Tekseriwdi júklew",
      "Новое исследование": "Jańa tekseriw",
      "Исследования": "Tekseriwler",
      // Фотопротокол. Термины взяты по аналогии с казахским и узбекским —
      // как и остальной словарь, требуют вычитки носителем.
      "Фотопротокол": "Fotoprotokol",
      "Лицо в покое": "Tınıshtaǵı bet",
      "Улыбка": "Jılmayıw",
      "Профиль": "Profil",
      "Прикус спереди": "Tislesiw aldınan",
      "Прикус справа": "Tislesiw oń jaqtan",
      "Прикус слева": "Tislesiw shep jaqtan",
      "Верхняя челюсть": "Joqarǵı jaq",
      "Нижняя челюсть": "Tómengi jaq",
      "Прочее": "Basqa",
      "Перетащите снимок или нажмите": "Súwretti súyreń yamasa basıń",
      "Загружаем…": "Júklenbekte…",
      "ещё": "taǵı",
      "Удалить снимок": "Súwretti óshiriw",
      "точно?": "anıq pa?",
      "снимок": "súwret",
      "снимка": "súwret",
      "снимков": "súwret",
      "снимков нет": "súwret joq",
      "Исследований пока нет": "Ele tekseriwler joq",
      "Что можно посчитать": "Nelerdi esaplaw múmkin",
      "Зубная карта": "Tis kartası",
      "находки со всех исследований": "barlıq tekseriwlerden tabılǵanlar",
      "есть находка": "tabılǵan bar",
      "зуб отсутствует": "tis joq",
      "Требует внимания": "Itibar talap etedi",
      "черновик": "jobası",
      "Собрать текстом": "Tekst retinde jıynaw",
      "Собираем…": "Jıynalmaqta…",
      "Сводка": "Juwmaq",
      "собрано моделью": "model jıynadı",
      "Противоречия между исследованиями": "Tekseriwler ortasındaǵı qarama-qarsılıqlar",
      "Чего не хватает для решения": "Sheshim ushın ne jetispeydi",
      "Загрузка…": "Júklenbekte…",
      "Обзор": "Sholıw",
      "Находки": "Tabılǵanları",
      "Находок нет": "Tabılǵan joq",
      "План лечения": "Emlew jobası",
      "Зубов": "Tisler",
      "с отметками": "belgili",
      "из": "ishinde",
      "находок": "tabılǵannan",
      "позиций в черновике": "jobadaǵı pozitsiyalar",
      "Показать все": "Barlıǵın kórsetiw",
      "на чём основано": "nege tiykarlanǵan",
      "свернуть": "jıynaw",
      "Считаем план…": "Joba esaplanbaqta…",
      "Плана нет: находок, ведущих к лечению, не найдено":
        "Joba joq: emlewge alıp baratuǵın tabılǵanlar tabılmadı",
      "Отброшено как не подтверждённое находками":
        "Tabılǵanlar menen tastıyıqlanbaǵanı ushın alıp taslandı",
      "Загрузить": "Júklew",
      "готово": "tayar",
      "считается": "esaplanbaqta",
      "в очереди": "gezekte",
      "загружено": "júklendi",
      "ошибка": "qátelik",
      "Готово": "Tayar",
      "Ошибка": "Qátelik",
      "Загрузка": "Júklenbekte",
      "Скачать отчёт": "Esabattı júklep alıw",
      "Отчёт": "Esabat",
      "Язык": "Til",
      // --- вкладки и панели ---
      "Находки": "Tabılǵanları",
      "План": "Joba",
      "План лечения": "Emlew jobası",
      "Срезы": "Kesimler",
      "Зубная карта": "Tis kartası",
      "Ручная проверка": "Qolda tekseriw",
      "Панорама": "Panorama",
      "Карточка зуба": "Tis kartochkası",
      // --- анатомия и слои ---
      "Челюсть": "Jaq",
      "Верхняя челюсть": "Joqarǵı jaq",
      "Нижняя челюсть": "Tómengi jaq",
      "Зубы": "Tisler",
      "Зуб": "Tis",
      "Нерв": "Nerv",
      "Пазухи": "Sinuslar",
      "Реставрации": "Restavraciyalar",
      "Дыхательные пути": "Demalıs jolları",
      "Десна": "Qızıl iyek",
      "Пульпа": "Pulpa",
      "Резцовый канал": "Kesiwshi kanal",
      "Язычный канал": "Til kanalı",
      // --- управление видом ---
      "Прозрачность": "Tunıqlıq",
      "Прозрачность десны": "Qızıl iyek tunıqlıǵı",
      "Блеск": "Jaltıraw",
      "Контакты": "Kontaktlar",
      "Раскраска": "Reńlew",
      "Естественная": "Tábiy",
      "Квадранты": "Kvadrantlar",
      "Скопировать ссылку на этот экран": "Bul ekranǵa siltemeni kóshiriw",
      "Интраоральный скан": "Awız ishi skanı",
      "Интеллектуальная диагностика панорамных снимков":
        "Panoramalıq súwretlerdi aqıllı diagnostikalaw",
      "Перетащите снимок сюда или нажмите для выбора":
        "Súwretti usı jerge taslań yamasa tańlaw ushın basıń",
      "Перетащите ZIP с DICOM-серией или файлы .dcm":
        "DICOM-seriyalı ZIP yamasa .dcm faylların taslań",
      "Или выберите пример из клинической базы DENTEX":
        "Yamasa DENTEX klinikalıq bazasınan mısal tańlań",
      "Или реальные снимки из нашей клиники":
        "Yamasa óz klinikamızdıń haqıyqıy súwretleri",
      "Или откройте готовый пример (без ожидания сегментации)":
        "Yamasa tayar mısaldı ashıń (segmentaciyanı kútpesten)",
      "Порог уверенности": "Isenimlilik shegi",
      "Сводка находок": "Tabılǵanlar juwmaǵı",
      "Найденные патологии": "Tabılǵan patologiyalar",
      "Находки и маршрут лечения": "Tabılǵanlar hám emlew baǵdarı",
      "← Новый снимок": "← Jańa súwret",
      "← Новый кейс": "← Jańa jaǵday",
      "Скопировать ссылку на этот кейс": "Bul jaǵdayǵa siltemeni kóshiriw",
      "Структуры": "Strukturalar",
      "Анализ ОПТГ за": "OPTG analizi",
      "доли секунды": "sekundtıń bólegi ishinde",
      "3D-анализ": "3D-analiz",
      "Нейросеть находит ретенированные зубы, кариес и периапикальные поражения и показывает их прямо на снимке": "Neyrotor retenirlengen tisler, kariyes hám periapikallıq zaqımlanıwlardı tabadı hám olardı súwrettiń ózinde kórsetedi",
      "Нейросеть сегментирует челюсти, зубы, нижний альвеолярный нерв и гайморовы пазухи — и строит интерактивную 3D-модель": "Neyrotor jaqlardı, tislerdi, tómengi alveolyar nervti hám gaymor sinuslardı bóledi hám interaktiv 3D-model quradı",
      "Панорамный снимок (ОПТГ) — нейросеть проанализирует его мгновенно":
        "Panoramalıq súwret (OPTG) — neyrotor onı tez arada analizleydi",
      "КЛКТ-исследование целиком — нейросеть выполнит сегментацию за 2–4 минуты":
        "Tolıq KLKT-tekseriwi — neyrotor segmentaciyanı 2–4 minutta orınlaydı",
      "Прозрачность кости и пазух": "Súyek hám sinuslar tunıqlıǵı",
      "Прозрачность зубов": "Tisler tunıqlıǵı",
      "Блеск зубов": "Tisler jaltıraǵı",
      "Карточка зуба": "Tis kartochkası",
      "Информация о кейсе": "Jaǵday haqqında maǵlıwmat",
      "3D-модель": "3D-model",
      "Панорама (ОПТГ)": "Panorama (OPTG)",
      "По зубам": "Tisler boyınsha",
      "Естественный": "Tábiy",
      "Показать": "Kórsetiw",
      "Скрыть": "Jasırıw",
      "Симметричная линия дуги": "Doganıń simmetriyalıq sızıǵı",
      "Челюсти и десна": "Jaqlar hám qızıl iyek",
      "Модель и слои": "Model hám qatlamlar",
      "Исходный скан": "Dáslepki skan",
      "Сегментация": "Segmentaciya",
      // --- роли врачей ---
      "Ортодонт": "Ortodont",
      "Ортопед": "Ortoped",
      "Хирург": "Xirurg",
      "Терапевт": "Terapevt",
      "Пародонтолог": "Parodontolog",
      "Гигиенист": "Gigienist",
      "Имплантолог": "Implantolog",
      // --- статусы проверки ---
      "Подтвердить": "Tastıyıqlaw",
      "Отклонить": "Biykarlaw",
      "Подписать": "Qol qoyıw",
      "Подписано": "Qol qoyılǵan",
      "Врач": "Shıpaker",
      "Проверено": "Tekserilgen",
    },
  };

  const KEY = "uiLang";
  /* Язык берётся из адреса, если он там указан: ссылку на экран врач
     пересылает целиком, и получатель должен увидеть тот же язык, а не свой
     прошлый выбор. Без параметра — прошлый выбор, по умолчанию русский. */
  const fromUrl = new URLSearchParams(location.search).get("lang");
  const KNOWN = ["ru", "kk", "uz"];
  let lang = KNOWN.indexOf(fromUrl) >= 0
    ? fromUrl : (localStorage.getItem(KEY) || "ru");
  if (lang !== "ru" && !DICT[lang]) lang = "ru";   // словаря нет — не делаем вид
  if (fromUrl) localStorage.setItem(KEY, lang);

  function t(s) {
    if (lang === "ru") return s;
    const d = DICT[lang];
    if (!d) return s;
    // Текст в разметке переносится по строкам, в словаре ключ записан одной
    // строкой: сравниваем по схлопнутым пробелам, иначе перенос в HTML молча
    // отключал бы перевод — и заметить это можно было бы только глазами.
    const v = d[String(s).replace(/\s+/g, " ").trim()];
    return v === undefined ? s : v;
  }

  /* Перевод разметки. Исходный русский держим в data-i18n-ru, иначе повторное
   * переключение переводило бы уже переведённый текст и портило его. */
  function apply(root) {
    (root || document).querySelectorAll("[data-i18n]").forEach(el => {
      if (el.dataset.i18nRu === undefined) el.dataset.i18nRu = el.textContent.trim();
      el.textContent = t(el.dataset.i18nRu);
    });
    (root || document).querySelectorAll("[data-i18n-title]").forEach(el => {
      if (el.dataset.i18nTitleRu === undefined) el.dataset.i18nTitleRu = el.title;
      el.title = t(el.dataset.i18nTitleRu);
    });
    document.documentElement.lang = lang === "kk" ? "kaa" : "ru";
  }

  let setLang = function (v) {
    lang = DICT[v] ? v : "ru";
    localStorage.setItem(KEY, lang);
    apply();
    document.dispatchEvent(new CustomEvent("langchange", { detail: lang }));
  };

  /* Переключатель. Ставится в контейнер с id="lang-switch", если он есть на
   * странице; иначе интерфейс просто остаётся русским. */
  function mount() {
    const box = document.getElementById("lang-switch");
    if (box) {
      box.innerHTML = "";
      /* RU / UZ — решение по рынку: клиника работает на русском и
         узбекском. Узбекского словаря ещё нет, и кнопка честно говорит об
         этом вместо того, чтобы молча оставить русский текст: «переключил,
         ничего не изменилось» читается как поломка.

         Каракалпакский словарь (DICT.kk) из файла НЕ УДАЛЁН: он написан и
         проверен, и переводится им обвязка, а не клинический текст.
         Включается адресом `?lang=kk` — так его можно показать, не ставя в
         шапку третью кнопку без перевода. */
      for (const [code, label] of [["ru", "RU"], ["uz", "UZ"]]) {
        const b = document.createElement("button");
        b.textContent = label;
        b.className = "lang-btn" + (code === lang ? " on" : "");
        b.onclick = () => {
          if (code === "uz" && !DICT.uz) {
            b.textContent = "скоро";
            setTimeout(() => { b.textContent = label; }, 1600);
            return;
          }
          setLang(code);
          box.querySelectorAll("button").forEach(x =>
            x.classList.toggle("on", x.textContent === (lang === "ru" ? "RU" : "UZ")));
        };
        box.appendChild(b);
      }
    }
    apply();
  }

  /* Подписка на смену языка. Нужна Vue-страницам: они рисуют разметку сами,
     и подмена textContent по data-i18n там не работает — при первой же
     перерисовке перевод затирается. Такие страницы зовут t() прямо в шаблоне
     и перерисовываются по этому событию. */
  const listeners = new Set();
  function onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }
  const _setLang = setLang;
  setLang = function (v) {
    _setLang(v);
    for (const fn of listeners) { try { fn(lang); } catch (e) {} }
  };

  window.i18n = { t, apply, setLang, mount, onChange,
                  get lang() { return lang; } };
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
