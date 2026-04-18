(function () {
  var nav = document.querySelector("[data-blog-nav]");
  var menu = document.querySelector("[data-blog-menu]");
  var menuToggle = document.querySelector("[data-blog-menu-toggle]");
  var backTop = document.querySelector("[data-blog-backtop]");
  var heroImage = document.querySelector(".blog-hero img");
  var lastScrollY = window.scrollY;

  function mixColor(start, end, amount) {
    return "rgb(" + start.map(function (value, index) {
      return Math.round(value + (end[index] - value) * amount);
    }).join(", ") + ")";
  }

  function updateChrome() {
    var current = window.scrollY;
    var max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
    var progress = Math.min(1, Math.max(0, current / max));

    if (nav) {
      nav.classList.toggle("is-top", current < 24);
      nav.classList.toggle("is-fixed", current >= 24);
      nav.classList.toggle("is-visible", current < lastScrollY || current < 24);
    }

    if (backTop) {
      var isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      var startColor = isDark ? [4, 51, 97] : [216, 237, 248];
      var endColor = isDark ? [2, 28, 56] : [126, 197, 232];
      backTop.style.setProperty("--scroll-progress", progress.toFixed(4));
      backTop.style.setProperty("--scroll-fill-color", mixColor(startColor, endColor, progress));
      backTop.classList.toggle("is-visible", current > 320);
      backTop.classList.toggle("is-past-half", progress >= 0.5);
      backTop.classList.toggle("is-dark", isDark);
    }

    if (heroImage) {
      heroImage.style.transform = "translate3d(0, " + (current * 0.16).toFixed(2) + "px, 0) scale(1.08)";
    }

    lastScrollY = current;
  }

  if (menuToggle && menu) {
    menuToggle.addEventListener("click", function () {
      menu.classList.toggle("is-open");
      menuToggle.classList.toggle("is-open");
    });
  }

  if (backTop) {
    backTop.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  function setupPagination() {
    var list = document.querySelector("[data-blog-paginated-list]");
    var pagination = document.querySelector("[data-blog-pagination]");
    if (!list || !pagination) return;

    var perPage = parseInt(list.getAttribute("data-page-size"), 10) || 25;
    var cards = Array.prototype.slice.call(list.querySelectorAll(".blog-card"));
    var totalPages = Math.ceil(cards.length / perPage);
    var currentPage = 1;

    if (totalPages <= 1) return;
    list.classList.add("is-ready");

    function scrollToFeedTop() {
      var target = list.getBoundingClientRect().top + window.pageYOffset - 88;
      window.scrollTo({ top: Math.max(0, target), behavior: "smooth" });
    }

    function render() {
      cards.forEach(function (card, index) {
        var page = Math.floor(index / perPage) + 1;
        card.hidden = page !== currentPage;
      });

      pagination.innerHTML = "";
      var previous = document.createElement("button");
      var indicator = document.createElement("span");
      var next = document.createElement("button");

      previous.type = "button";
      previous.className = "blog-page-previous";
      previous.textContent = "Previous";
      previous.disabled = currentPage === 1;

      indicator.className = "blog-page-indicator";
      indicator.textContent = currentPage + " / " + totalPages;

      next.type = "button";
      next.className = "blog-page-next";
      next.textContent = "Next";
      next.disabled = currentPage === totalPages;

      previous.addEventListener("click", function () {
        if (currentPage > 1) {
          currentPage -= 1;
          render();
          scrollToFeedTop();
        }
      });

      next.addEventListener("click", function () {
        if (currentPage < totalPages) {
          currentPage += 1;
          render();
          scrollToFeedTop();
        }
      });

      pagination.appendChild(previous);
      pagination.appendChild(indicator);
      pagination.appendChild(next);
    }

    render();
  }

  function setupTagExpansion() {
    var tagCloud = document.querySelector("[data-blog-tags-collapsed]");
    var toggle = document.querySelector("[data-blog-tags-toggle]");
    if (!tagCloud || !toggle) return;

    function sync() {
      var expanded = tagCloud.classList.contains("is-expanded");
      toggle.textContent = expanded ? "Show less" : "Show all";
    }

    toggle.addEventListener("click", function () {
      tagCloud.classList.toggle("is-expanded");
      sync();
    });

    sync();
  }

  function setupSearch() {
    var overlay = document.querySelector("[data-blog-search]");
    var openers = Array.prototype.slice.call(document.querySelectorAll("[data-blog-search-open]"));
    var close = document.querySelector("[data-blog-search-close]");
    var input = document.querySelector("[data-blog-search-input]");
    var results = document.querySelector("[data-blog-search-results]");
    var index = null;
    var closeTimer = null;

    if (!overlay || !input || !results || !openers.length) return;

    function render(items) {
      if (!input.value.trim()) {
        results.innerHTML = "<p>Type to search posts, subtitles, descriptions, and tags.</p>";
        return;
      }

      if (!items.length) {
        results.innerHTML = "<p>No matching posts.</p>";
        return;
      }

      results.innerHTML = items.slice(0, 20).map(function (item) {
        var title = escapeHtml(item.title);
        var summary = escapeHtml(item.subtitle || item.description || item.tags || "");
        return [
          '<a class="blog-search-result" href="', encodeURI(item.url), '">',
          '<time>', escapeHtml(item.date), '</time>',
          '<strong>', title, '</strong>',
          '<span>', summary, '</span>',
          '</a>'
        ].join("");
      }).join("");
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, function (char) {
        return {
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;"
        }[char];
      });
    }

    function search() {
      var query = input.value.trim().toLowerCase();
      if (!query || !index) {
        render([]);
        return;
      }

      var words = query.split(/\s+/);
      var matched = index.filter(function (item) {
        var haystack = [item.title, item.subtitle, item.description, item.tags, item.date].join(" ").toLowerCase();
        return words.every(function (word) {
          return haystack.indexOf(word) !== -1;
        });
      });
      render(matched);
    }

    function openSearch() {
      if (closeTimer) {
        window.clearTimeout(closeTimer);
        closeTimer = null;
      }
      overlay.hidden = false;
      window.requestAnimationFrame(function () {
        overlay.classList.add("is-open");
      });
      document.documentElement.classList.add("blog-search-open");
      input.focus();
      if (!index) {
        fetch("/blog/search.json")
          .then(function (response) { return response.json(); })
          .then(function (data) {
            index = data;
            search();
          })
          .catch(function () {
            results.innerHTML = "<p>Search index failed to load.</p>";
          });
      } else {
        search();
      }
    }

    function closeSearch() {
      overlay.classList.remove("is-open");
      document.documentElement.classList.remove("blog-search-open");
      closeTimer = window.setTimeout(function () {
        overlay.hidden = true;
      }, 180);
    }

    openers.forEach(function (opener) {
      opener.addEventListener("click", openSearch);
    });
    close.addEventListener("click", closeSearch);
    input.addEventListener("input", search);
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && !overlay.hidden) closeSearch();
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        openSearch();
      }
    });

    render([]);
  }

  function buildToc() {
    var toc = document.querySelector("[data-blog-toc]");
    var tocBody = document.querySelector("[data-blog-toc-body]");
    var content = document.querySelector(".blog-content");
    if (!toc || !tocBody || !content) return;

    var headings = Array.prototype.slice.call(content.querySelectorAll("h1, h2, h3, h4"));
    if (!headings.length) {
      toc.style.display = "none";
      return;
    }

    headings.forEach(function (heading, index) {
      if (!heading.id) {
        heading.id = "heading-" + index;
      }

      var item = document.createElement("li");
      item.className = "toc-" + heading.tagName.toLowerCase();
      var link = document.createElement("a");
      link.href = "#" + heading.id;
      link.textContent = heading.textContent;
      item.appendChild(link);
      tocBody.appendChild(item);
    });

    var links = Array.prototype.slice.call(tocBody.querySelectorAll("a"));
    if ("IntersectionObserver" in window) {
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          links.forEach(function (link) {
            link.classList.toggle("is-active", link.getAttribute("href") === "#" + entry.target.id);
          });
        });
      }, { rootMargin: "-20% 0px -70% 0px", threshold: 0.01 });

      headings.forEach(function (heading) {
        observer.observe(heading);
      });
    }

    var toggle = document.querySelector("[data-blog-toc-toggle]");
    if (toggle) {
      toggle.addEventListener("click", function () {
        toc.classList.toggle("is-folded");
      });
    }
  }

  function setupArchiveFilter() {
    var tagBox = document.querySelector(".js-tags");
    var results = document.querySelector(".js-archive-results");
    if (!tagBox || !results) return;

    var buttons = Array.prototype.slice.call(tagBox.querySelectorAll("[data-tag]"));
    var items = Array.prototype.slice.call(results.querySelectorAll("[data-tags]"));

    function normalizeTag(tag) {
      if (!tag) return "";
      if (tag.indexOf("%25") !== -1) {
        try {
          tag = decodeURIComponent(tag);
        } catch (error) {
          return tag;
        }
      }
      if (tag.indexOf("%") === -1) {
        return encodeURIComponent(tag).replace(/%20/g, "+");
      }
      try {
        return encodeURIComponent(decodeURIComponent(tag)).replace(/%20/g, "+");
      } catch (error) {
        return tag;
      }
    }

    function filter(tag) {
      tag = normalizeTag(tag);
      buttons.forEach(function (button) {
        button.classList.toggle("is-active", button.getAttribute("data-tag") === tag);
      });

      items.forEach(function (item) {
        var tags = (item.getAttribute("data-tags") || "").split(",").filter(Boolean);
        item.hidden = Boolean(tag) && tags.indexOf(tag) === -1;
      });

      Array.prototype.slice.call(results.querySelectorAll(".blog-archive-year")).forEach(function (year) {
        var visible = Array.prototype.slice.call(year.querySelectorAll("[data-tags]")).some(function (item) {
          return !item.hidden;
        });
        year.hidden = !visible;
      });

      var next = window.location.pathname + (tag ? "?tag=" + tag : "");
      window.history.replaceState({}, "", next);
    }

    function rawQueryTag() {
      var query = window.location.search.replace(/^\?/, "").split("&");
      for (var i = 0; i < query.length; i += 1) {
        var pair = query[i].split("=");
        if (pair[0] === "tag") return pair.slice(1).join("=");
      }
      return "";
    }

    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        filter(button.getAttribute("data-tag"));
      });
    });

    filter(rawQueryTag());
  }

  window.addEventListener("scroll", updateChrome, { passive: true });
  updateChrome();
  buildToc();
  setupArchiveFilter();
  setupPagination();
  setupTagExpansion();
  setupSearch();
})();
