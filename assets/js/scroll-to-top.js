// Scroll to top button with progress indicator
(function() {
  'use strict';

  // Create the button element
  const scrollBtn = document.createElement('div');
  scrollBtn.id = 'scroll-to-top';
  scrollBtn.innerHTML = `
    <svg class="progress-ring" width="60" height="60">
      <circle class="progress-ring-circle" stroke="#043361" stroke-width="3" fill="transparent" r="26" cx="30" cy="30"/>
    </svg>
    <i class="fas fa-arrow-up"></i>
  `;
  document.body.appendChild(scrollBtn);

  const progressCircle = scrollBtn.querySelector('.progress-ring-circle');
  const radius = progressCircle.r.baseVal.value;
  const circumference = radius * 2 * Math.PI;
  
  progressCircle.style.strokeDasharray = `${circumference} ${circumference}`;
  progressCircle.style.strokeDashoffset = circumference;

  // Update progress on scroll
  function updateProgress() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrollPercent = scrollTop / scrollHeight;
    
    const offset = circumference - scrollPercent * circumference;
    progressCircle.style.strokeDashoffset = offset;

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

  // Update on scroll
  window.addEventListener('scroll', updateProgress);
  
  // Initial update
  updateProgress();
})();
