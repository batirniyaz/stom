/* Вход врача: сессия и защита экранов.
 *
 * ЧЕСТНО О ТОМ, ЧТО ЭТО. Настоящей аутентификации здесь нет: кода из SMS
 * никто не шлёт, пароль не проверяется, сервер о сессии не знает. Это
 * ДЕМОНСТРАЦИОННЫЙ вход — он показывает, как выглядит работа врача со
 * своей учётной записью, и не защищает данные. Любой, кто откроет адрес,
 * войдёт по коду, который написан тут же на экране.
 *
 * Почему тогда вообще сделан. Сценарий показа начинается с главной, и без
 * входа непонятно, чьи это пациенты и откуда в отчёте берётся фамилия
 * врача. Настоящую проверку делает сервер клиники, и когда она появится,
 * этот файл меняется на запрос к ней — экраны и переходы остаются теми же.
 *
 * Гостевая ссылка (`?guest=1`) пускает без входа и прячет кнопки проверки:
 * ею делятся с коллегой, чтобы он ПОСМОТРЕЛ снимок, а не подписал его.
 * Это тоже не ограничение доступа, а подсказка интерфейса.
 */
(function () {
  const KEY = "actina_session";

  function read() {
    try {
      return JSON.parse(localStorage.getItem(KEY) || sessionStorage.getItem(KEY) || "null");
    } catch (e) { return null; }
  }

  function start(phone, remember) {
    const s = { phone: phone || "", at: Date.now() };
    const store = remember ? localStorage : sessionStorage;
    try { store.setItem(KEY, JSON.stringify(s)); } catch (e) {}
    // Если врач раньше просил запомнить компьютер, а теперь снял галочку —
    // старую запись надо убрать, иначе «не запоминать» ничего не меняет.
    if (!remember) { try { localStorage.removeItem(KEY); } catch (e) {} }
    return s;
  }

  function end() {
    try { localStorage.removeItem(KEY); sessionStorage.removeItem(KEY); } catch (e) {}
  }

  function isGuest() {
    return new URLSearchParams(location.search).get("guest") === "1";
  }

  /* Защита экрана. Вызывается в <head> страницы, ДО отрисовки: иначе врач
     на долю секунды видит чужой экран, а на показе это читается как сбой.
     Возврат запоминаем, чтобы после входа попасть туда, куда шли. */
  function guard() {
    if (read() || isGuest()) return true;
    try { sessionStorage.setItem("actina_after", location.pathname + location.search); } catch (e) {}
    location.replace("/?login=1");
    return false;
  }

  function afterLogin() {
    let t = null;
    try { t = sessionStorage.getItem("actina_after"); sessionStorage.removeItem("actina_after"); } catch (e) {}
    return t || "/patients";
  }

  window.actinaAuth = { read, start, end, guard, isGuest, afterLogin, KEY };
})();
