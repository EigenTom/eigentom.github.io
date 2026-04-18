(function () {
  function triggerKey(keyCode, type) {
    var canvas = document.getElementById("canvas");
    if (!canvas) return;
    var eventObj = document.createEventObject ? document.createEventObject() : document.createEvent("Events");
    if (eventObj.initEvent) eventObj.initEvent(type, true, true);
    eventObj.keyCode = keyCode;
    eventObj.which = keyCode;
    if (canvas.dispatchEvent) {
      canvas.dispatchEvent(eventObj);
    } else {
      canvas.fireEvent("on" + type, eventObj);
    }
  }

  function dispatchTextKey(char) {
    var canvas = document.getElementById("canvas");
    if (!canvas) return;
    var isEnter = char === "\n";
    var key = isEnter ? "Enter" : char;
    var keyCode = isEnter ? 13 : char.toUpperCase().charCodeAt(0);
    var events = ["keydown", "keypress", "keyup"];
    canvas.focus();

    events.forEach(function (type) {
      var eventObj;
      try {
        eventObj = new KeyboardEvent(type, {
          key: key,
          code: isEnter ? "Enter" : "",
          keyCode: keyCode,
          which: keyCode,
          charCode: type === "keypress" ? keyCode : 0,
          bubbles: true,
          cancelable: true
        });
      } catch (error) {
        eventObj = document.createEvent("Events");
        eventObj.initEvent(type, true, true);
      }
      try {
        Object.defineProperty(eventObj, "keyCode", { get: function () { return keyCode; } });
        Object.defineProperty(eventObj, "which", { get: function () { return keyCode; } });
      } catch (error) {}
      canvas.dispatchEvent(eventObj);
    });
  }

  function requestPageFullscreen() {
    var target = document.documentElement;
    if (document.fullscreenElement) {
      document.exitFullscreen();
      return;
    }
    if (target.requestFullscreen) target.requestFullscreen();
  }

  function requestEmulatorFullscreen() {
    var resize = document.getElementById("resize");
    var pointerLock = document.getElementById("pointerLock");
    if (window.Module && typeof window.Module.requestFullscreen === "function") {
      window.Module.requestFullscreen(Boolean(pointerLock && pointerLock.checked), Boolean(resize && resize.checked));
    } else {
      requestPageFullscreen();
    }
  }

  function setupWindow() {
    var shell = document.querySelector(".touhou-shell");
    var win = document.querySelector(".win95-window");
    var bar = document.querySelector(".win95-titlebar");
    if (!shell || !win || !bar) return;

    var locked = false;
    var dragging = false;
    var offsetX = 0;
    var offsetY = 0;

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function startDrag(event) {
      if (locked || event.target.closest(".win95-controls")) return;
      dragging = true;
      var rect = win.getBoundingClientRect();
      offsetX = event.clientX - rect.left;
      offsetY = event.clientY - rect.top;
      win.classList.add("is-floating");
      win.style.left = rect.left + "px";
      win.style.top = rect.top + "px";
      win.style.width = rect.width + "px";
      shell.style.minHeight = Math.max(shell.offsetHeight, rect.height + 32) + "px";
      bar.setPointerCapture(event.pointerId);
      event.preventDefault();
    }

    function drag(event) {
      if (!dragging) return;
      var rect = win.getBoundingClientRect();
      var left = clamp(event.clientX - offsetX, 8, window.innerWidth - rect.width - 8);
      var top = clamp(event.clientY - offsetY, 8, window.innerHeight - rect.height - 8);
      win.style.left = left + "px";
      win.style.top = top + "px";
    }

    function stopDrag(event) {
      if (!dragging) return;
      dragging = false;
      try {
        bar.releasePointerCapture(event.pointerId);
      } catch (error) {}
    }

    bar.addEventListener("pointerdown", startDrag);
    bar.addEventListener("pointermove", drag);
    bar.addEventListener("pointerup", stopDrag);
    bar.addEventListener("pointercancel", stopDrag);

    var lockButton = document.querySelector("[data-win-lock]");
    if (lockButton) {
      lockButton.addEventListener("click", function () {
        locked = !locked;
        win.classList.toggle("is-locked", locked);
      });
    }

    var fullButton = document.querySelector("[data-win-fullscreen]");
    if (fullButton) fullButton.addEventListener("click", requestEmulatorFullscreen);

    var closeButton = document.querySelector("[data-win-close]");
    if (closeButton) {
      closeButton.addEventListener("click", function () {
        window.location.href = "../main.html#touhou";
      });
    }

    setupWindowResize(win);
  }

  function makeDraggable(win, handle) {
    if (!win || !handle) return;
    var dragging = false;
    var offsetX = 0;
    var offsetY = 0;

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    handle.addEventListener("pointerdown", function (event) {
      if (event.target.closest("button")) return;
      dragging = true;
      var rect = win.getBoundingClientRect();
      offsetX = event.clientX - rect.left;
      offsetY = event.clientY - rect.top;
      win.style.left = rect.left + "px";
      win.style.top = rect.top + "px";
      win.style.right = "auto";
      win.style.bottom = "auto";
      handle.setPointerCapture(event.pointerId);
      event.preventDefault();
    });

    handle.addEventListener("pointermove", function (event) {
      if (!dragging) return;
      var rect = win.getBoundingClientRect();
      win.style.left = clamp(event.clientX - offsetX, 8, window.innerWidth - rect.width - 8) + "px";
      win.style.top = clamp(event.clientY - offsetY, 8, window.innerHeight - rect.height - 8) + "px";
      event.preventDefault();
    });

    function stop(event) {
      if (!dragging) return;
      dragging = false;
      try {
        handle.releasePointerCapture(event.pointerId);
      } catch (error) {}
    }

    handle.addEventListener("pointerup", stop);
    handle.addEventListener("pointercancel", stop);
  }

  function setupWindowResize(win) {
    var grip = document.querySelector("[data-window-resize]");
    var canvas = document.getElementById("canvas");
    if (!grip || !canvas) return;

    var resizing = false;
    var startX = 0;
    var startWidth = 0;

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function currentCanvasWidth() {
      return canvas.getBoundingClientRect().width || canvas.width || 900;
    }

    function start(event) {
      resizing = true;
      startX = event.clientX;
      startWidth = currentCanvasWidth();
      document.body.classList.add("canvas-resize");
      grip.setPointerCapture(event.pointerId);
      event.preventDefault();
    }

    function move(event) {
      if (!resizing) return;
      var maxWidth = Math.max(360, window.innerWidth - 80);
      var nextWidth = clamp(startWidth + event.clientX - startX, 420, maxWidth);
      document.documentElement.style.setProperty("--touhou-canvas-width", nextWidth.toFixed(0) + "px");
      if (win.classList.contains("is-floating")) {
        win.style.width = "";
      }
      event.preventDefault();
    }

    function stop(event) {
      if (!resizing) return;
      resizing = false;
      try {
        grip.releasePointerCapture(event.pointerId);
      } catch (error) {}
      event.preventDefault();
    }

    grip.addEventListener("pointerdown", start);
    grip.addEventListener("pointermove", move);
    grip.addEventListener("pointerup", stop);
    grip.addEventListener("pointercancel", stop);
  }

  function setupJoystick() {
    var activeKeys = {};
    var repeatTimers = {};
    var repeatDelay = 260;
    var repeatInterval = 170;

    function setKey(keyCode, isDown) {
      if (isDown && !activeKeys[keyCode]) {
        activeKeys[keyCode] = true;
        triggerKey(keyCode, "keydown");
        return;
      }
      if (!isDown && activeKeys[keyCode]) {
        activeKeys[keyCode] = false;
        triggerKey(keyCode, "keyup");
      }
    }

    function startRepeat(keyCode) {
      stopRepeat(keyCode);
      repeatTimers[keyCode] = window.setTimeout(function tick() {
        if (activeKeys[keyCode]) {
          triggerKey(keyCode, "keydown");
          repeatTimers[keyCode] = window.setTimeout(tick, repeatInterval);
        }
      }, repeatDelay);
    }

    function stopRepeat(keyCode) {
      if (repeatTimers[keyCode]) {
        window.clearTimeout(repeatTimers[keyCode]);
        repeatTimers[keyCode] = null;
      }
    }

    function holdKey(keyCode, isDown) {
      var wasDown = Boolean(activeKeys[keyCode]);
      setKey(keyCode, isDown);
      if (isDown && !wasDown) startRepeat(keyCode);
      if (!isDown) stopRepeat(keyCode);
    }

    function releaseKeys(keyCodes) {
      keyCodes.forEach(function (keyCode) {
        holdKey(keyCode, false);
      });
    }

    Array.prototype.slice.call(document.querySelectorAll("[data-key]")).forEach(function (button) {
      var keyCode = parseInt(button.getAttribute("data-key"), 10);

      function down(event) {
        holdKey(keyCode, true);
        button.classList.add("is-active");
        event.preventDefault();
      }

      function up(event) {
        holdKey(keyCode, false);
        button.classList.remove("is-active");
        event.preventDefault();
      }

      button.addEventListener("pointerdown", down);
      button.addEventListener("pointerup", up);
      button.addEventListener("pointerleave", up);
      button.addEventListener("pointercancel", up);
    });

    var joystick = document.querySelector("[data-joystick]");
    var knob = document.querySelector("[data-joystick-knob]");
    if (!joystick || !knob) return;

    var directionKeys = [37, 38, 39, 40];
    var joystickActive = false;
    var radius = 46;
    var threshold = 24;
    var diagonalThreshold = 32;

    function applyJoystick(clientX, clientY) {
      var rect = joystick.getBoundingClientRect();
      var centerX = rect.left + rect.width / 2;
      var centerY = rect.top + rect.height / 2;
      var dx = clientX - centerX;
      var dy = clientY - centerY;
      var distance = Math.min(radius, Math.sqrt(dx * dx + dy * dy));
      var angle = Math.atan2(dy, dx);
      var x = Math.cos(angle) * distance;
      var y = Math.sin(angle) * distance;
      var nextKeys = {};
      var absX = Math.abs(x);
      var absY = Math.abs(y);

      knob.style.transform = "translate(" + x.toFixed(1) + "px, " + y.toFixed(1) + "px)";

      if (distance >= threshold) {
        if (absX >= absY * 0.58 && absX >= threshold) nextKeys[x < 0 ? 37 : 39] = true;
        if (absY >= absX * 0.58 && absY >= threshold) nextKeys[y < 0 ? 38 : 40] = true;
        if (distance < diagonalThreshold && absX !== absY) {
          if (absX > absY) {
            delete nextKeys[38];
            delete nextKeys[40];
          } else {
            delete nextKeys[37];
            delete nextKeys[39];
          }
        }
      }

      directionKeys.forEach(function (keyCode) {
        holdKey(keyCode, Boolean(nextKeys[keyCode]));
      });
    }

    function startJoystick(event) {
      joystickActive = true;
      joystick.classList.add("is-active");
      joystick.setPointerCapture(event.pointerId);
      applyJoystick(event.clientX, event.clientY);
      event.preventDefault();
    }

    function moveJoystick(event) {
      if (!joystickActive) return;
      applyJoystick(event.clientX, event.clientY);
      event.preventDefault();
    }

    function stopJoystick(event) {
      if (!joystickActive) return;
      joystickActive = false;
      knob.style.transform = "translate(0, 0)";
      joystick.classList.remove("is-active");
      releaseKeys(directionKeys);
      try {
        joystick.releasePointerCapture(event.pointerId);
      } catch (error) {}
      event.preventDefault();
    }

    joystick.addEventListener("pointerdown", startJoystick);
    joystick.addEventListener("pointermove", moveJoystick);
    joystick.addEventListener("pointerup", stopJoystick);
    joystick.addEventListener("pointercancel", stopJoystick);

    var dpad = document.createElement("div");
    dpad.className = "dpad";
    dpad.setAttribute("aria-label", "Directional buttons");
    dpad.innerHTML = [
      '<button class="arrow_keys dpad-up" type="button" data-dpad-key="38">↑</button>',
      '<button class="arrow_keys dpad-left" type="button" data-dpad-key="37">←</button>',
      '<span class="dpad-center" aria-hidden="true"></span>',
      '<button class="arrow_keys dpad-right" type="button" data-dpad-key="39">→</button>',
      '<button class="arrow_keys dpad-down" type="button" data-dpad-key="40">↓</button>'
    ].join("");
    joystick.parentNode.insertBefore(dpad, joystick.nextSibling);

    Array.prototype.slice.call(dpad.querySelectorAll("[data-dpad-key]")).forEach(function (button) {
      var keyCode = parseInt(button.getAttribute("data-dpad-key"), 10);

      function down(event) {
        holdKey(keyCode, true);
        button.classList.add("is-active");
        event.preventDefault();
      }

      function up(event) {
        holdKey(keyCode, false);
        button.classList.remove("is-active");
        event.preventDefault();
      }

      button.addEventListener("pointerdown", down);
      button.addEventListener("pointerup", up);
      button.addEventListener("pointerleave", up);
      button.addEventListener("pointercancel", up);
    });
  }

  function setupResizeToggle() {
    var resize = document.getElementById("resize");
    if (!resize) return;

    function sync() {
      document.body.classList.toggle("canvas-resize", resize.checked);
    }

    resize.addEventListener("change", sync);
    sync();
  }

  function setupControlPanel() {
    var controls = document.getElementById("controls");
    if (!controls) return;

    var mode = document.createElement("label");
    mode.innerHTML = 'Control <select data-control-mode><option value="joystick">Joystick</option><option value="dpad">Arrow Keys</option></select>';
    controls.appendChild(mode);

    var logButton = document.createElement("button");
    logButton.type = "button";
    logButton.className = "win95-mini-button";
    logButton.textContent = "Log";
    controls.appendChild(logButton);

    var command = document.createElement("form");
    command.className = "dos-command";
    command.innerHTML = '<input type="text" data-dos-command placeholder="DOS input"><button type="submit">Send</button>';
    controls.appendChild(command);

    var select = mode.querySelector("[data-control-mode]");
    select.addEventListener("change", function () {
      document.body.classList.toggle("dpad-mode", select.value === "dpad");
    });

    logButton.addEventListener("click", function () {
      document.body.classList.toggle("show-log");
    });

    command.addEventListener("submit", function (event) {
      event.preventDefault();
      var input = command.querySelector("[data-dos-command]");
      var text = input.value;
      if (!text) return;
      (text + "\n").split("").forEach(dispatchTextKey);
      input.value = "";
    });
  }

  function setupLogWindow() {
    var source = document.getElementById("output");
    var win = document.createElement("section");
    win.className = "win95-log-window";
    win.innerHTML = [
      '<div class="win95-titlebar win95-log-titlebar">',
      '<span>DOSBOX.LOG</span>',
      '<span class="win95-controls"><button type="button" data-log-close title="Hide log">x</button></span>',
      '</div>',
      '<textarea class="dos-log-output" readonly spellcheck="false"></textarea>'
    ].join("");
    document.body.appendChild(win);

    var textarea = win.querySelector(".dos-log-output");
    var titlebar = win.querySelector(".win95-log-titlebar");
    var close = win.querySelector("[data-log-close]");
    makeDraggable(win, titlebar);

    function append(line) {
      textarea.value += String(line) + "\n";
      textarea.scrollTop = textarea.scrollHeight;
      if (source) {
        source.value += String(line) + "\n";
        source.scrollTop = source.scrollHeight;
      }
    }

    close.addEventListener("click", function () {
      document.body.classList.remove("show-log");
    });

    var originalLog = console.log;
    var originalError = console.error;
    console.log = function () {
      append(Array.prototype.slice.call(arguments).join(" "));
      originalLog.apply(console, arguments);
    };
    console.error = function () {
      append("[error] " + Array.prototype.slice.call(arguments).join(" "));
      originalError.apply(console, arguments);
    };

    if (window.Module && typeof window.Module.setStatus === "function") {
      var originalStatus = window.Module.setStatus;
      window.Module.setStatus = function (text) {
        if (text) append("[status] " + text);
        return originalStatus.apply(window.Module, arguments);
      };
    }
  }

  window.addEventListener("DOMContentLoaded", function () {
    setupWindow();
    setupJoystick();
    setupResizeToggle();
    setupControlPanel();
    setupLogWindow();
  });
})();
