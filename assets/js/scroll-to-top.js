// Scroll to top button with progress indicator
(function() {
  'use strict';

  // Create the button element
  const scrollBtn = document.createElement('div');
  scrollBtn.id = 'scroll-to-top';
  scrollBtn.innerHTML = `
    <span class="scroll-fill" aria-hidden="true"></span>
    <i class="fas fa-arrow-up"></i>
  `;
  scrollBtn.setAttribute('role', 'button');
  scrollBtn.setAttribute('aria-label', 'Scroll to top');
  scrollBtn.setAttribute('tabindex', '0');
  document.body.appendChild(scrollBtn);

  function mixColor(start, end, amount) {
    const mixed = start.map((value, index) => {
      return Math.round(value + (end[index] - value) * amount);
    });

    return `rgb(${mixed[0]}, ${mixed[1]}, ${mixed[2]})`;
  }

  // Update progress on scroll
  function updateProgress() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrollPercent = scrollHeight > 0 ? Math.min(scrollTop / scrollHeight, 1) : 0;
    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const startColor = isDark ? [4, 51, 97] : [216, 237, 248];
    const endColor = isDark ? [2, 28, 56] : [126, 197, 232];

    scrollBtn.style.setProperty('--scroll-progress', scrollPercent.toFixed(4));
    scrollBtn.style.setProperty('--scroll-fill-color', mixColor(startColor, endColor, scrollPercent));
    scrollBtn.classList.toggle('is-dark', isDark);
    scrollBtn.classList.toggle('is-past-half', !isDark && scrollPercent >= 0.5);

    // Show/hide button based on scroll position
    if (scrollTop > 50) {
      scrollBtn.classList.add('visible');
    } else {
      scrollBtn.classList.remove('visible');
    }
  }

  // Scroll to top on click
  scrollBtn.addEventListener('click', function() {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });

  scrollBtn.addEventListener('keydown', function(event) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      scrollBtn.click();
    }
  });

  // Update on scroll
  window.addEventListener('scroll', updateProgress);

  // Initial update
  updateProgress();
})();
