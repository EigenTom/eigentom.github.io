(function () {
  var nav = document.querySelector(".project-nav");
  var backTop = document.querySelector("[data-project-backtop]");

  function updateChrome() {
    var current = window.scrollY || document.documentElement.scrollTop;
    if (nav) nav.classList.toggle("is-solid", current > 40);
    if (backTop) backTop.hidden = current < 420;
  }

  if (backTop) {
    backTop.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  window.addEventListener("scroll", updateChrome, { passive: true });
  updateChrome();
})();
