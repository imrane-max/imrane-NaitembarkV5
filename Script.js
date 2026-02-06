// ===========================
// Script - interactive enhancements
// - theme persistence + keyboard toggle
// - scroll progress bar
// - reduced-motion respect
// - improved copy-email toast
// - retains existing animations/stagger behavior
// ===========================

const themeBtn = document.getElementById("ThemBtn");
const textEmail = "imrane2015su@gmial.com";

// Respect user's reduced-motion preference
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Apply saved theme on load
function applySavedTheme() {
  const saved = localStorage.getItem('site-theme');
  if (saved === 'dark') {
    document.body.classList.add('dark-mode');
    themeBtn.textContent = 'Light';
  } else {
    document.body.classList.remove('dark-mode');
    themeBtn.textContent = 'Dark';
  }
}

applySavedTheme();

// Toggle theme helper
function toggleTheme(save = true) {
  const nowDark = document.body.classList.toggle('dark-mode');
  themeBtn.textContent = nowDark ? 'Light' : 'Dark';
  themeBtn.classList.add('glow-animation');
  setTimeout(() => themeBtn.classList.remove('glow-animation'), 2000);
  if (save) localStorage.setItem('site-theme', nowDark ? 'dark' : 'light');
}

themeBtn.addEventListener('click', () => toggleTheme(true));

// Keyboard shortcut: 'D' toggles theme
window.addEventListener('keydown', (e) => {
  if (e.key.toLowerCase() === 'd' && !e.metaKey && !e.ctrlKey && !e.altKey) {
    toggleTheme(true);
  }
});

// ===========================
// Fade-in Animation on Scroll (IntersectionObserver)
// ===========================
const fadeElements = document.querySelectorAll('.fade-in');
if (!prefersReducedMotion) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) entry.target.classList.add('visible');
    });
  }, { threshold: 0.2 });
  fadeElements.forEach(el => observer.observe(el));
} else {
  // If reduced motion, make all visible immediately
  fadeElements.forEach(el => el.classList.add('visible'));
}

// ===========================
// Toast utility for user feedback
// ===========================
function showToast(message, duration = 2000) {
  const toast = document.createElement('div');
  toast.className = 'site-toast';
  toast.textContent = message;
  document.body.appendChild(toast);
  // basic styles if not present in CSS
  toast.style.position = 'fixed';
  toast.style.bottom = '24px';
  toast.style.left = '50%';
  toast.style.transform = 'translateX(-50%)';
  toast.style.background = 'rgba(0,0,0,0.8)';
  toast.style.color = '#fff';
  toast.style.padding = '10px 16px';
  toast.style.borderRadius = '8px';
  toast.style.zIndex = 9999;
  toast.style.fontWeight = '600';
  toast.style.opacity = '0';
  toast.style.transition = 'opacity 0.25s ease';
  requestAnimationFrame(() => (toast.style.opacity = '1'));
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.addEventListener('transitionend', () => toast.remove());
  }, duration);
}

// ===========================
// Copy Email with fallback and toast
// ===========================
function copyEmail(event) {
  const caller = (event && event.target) ? event.target.closest('button') : null;
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(textEmail).then(() => {
      if (caller) {
        const original = caller.innerHTML;
        caller.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
        caller.classList.add('glow-animation');
        setTimeout(() => {
          caller.innerHTML = original;
          caller.classList.remove('glow-animation');
        }, 1800);
      }
      showToast('Email copied to clipboard');
    }).catch(() => {
      // fallback
      showToast('Copy failed — select and copy manually');
      if (caller) caller.focus();
    });
  } else {
    // older fallback: prompt
    try {
      window.prompt('Copy email:', textEmail);
    } catch (e) {
      showToast('Unable to copy email');
    }
  }
}

// Expose function to global scope for inline onclick usage
window.copyEmail = copyEmail;

// ===========================
// Card hover float animation (if not reduced motion)
// ===========================
const cards = document.querySelectorAll('.card');
cards.forEach((card) => {
  if (!prefersReducedMotion) {
    card.addEventListener('mouseenter', () => {
      card.style.animation = `float 0.6s ease-in-out forwards`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.animation = 'none';
    });
  }
});

// ===========================
// Smooth Scroll for Navigation Links
// ===========================
const navLinks = document.querySelectorAll('.navbar a[href^="#"]');
navLinks.forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const targetId = link.getAttribute('href');
    const targetSection = document.querySelector(targetId);
    if (targetSection) {
      targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// ===========================
// Stagger animations for skills and project cards
// ===========================
const skillSpans = document.querySelectorAll('.skills span');
skillSpans.forEach((skill, index) => {
  skill.style.animationDelay = `${index * 0.1}s`;
  skill.classList.add('slide-left');
});

const projectCards = document.querySelectorAll('.cards .card');
projectCards.forEach((card, index) => {
  card.style.animationDelay = `${index * 0.15}s`;
  if (index % 2 === 0) card.classList.add('slide-left'); else card.classList.add('slide-right');
});

// ===========================
// Button ripple mouse tracking
// ===========================
const buttons = document.querySelectorAll('.btn');
buttons.forEach(btn => {
  btn.addEventListener('mousemove', (e) => {
    const rect = btn.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    btn.style.setProperty('--mouse-x', `${x}px`);
    btn.style.setProperty('--mouse-y', `${y}px`);
  });
});

// ===========================
// Scroll progress bar
// ===========================
function createProgressBar() {
  const bar = document.createElement('div');
  bar.id = 'scroll-progress';
  bar.style.position = 'fixed';
  bar.style.top = '0';
  bar.style.left = '0';
  bar.style.height = '4px';
  bar.style.width = '0%';
  bar.style.zIndex = '9999';
  bar.style.background = 'linear-gradient(90deg, #007bff, #00d4ff)';
  bar.style.transition = 'width 0.15s linear';
  document.body.appendChild(bar);
  return bar;
}

const progressBar = createProgressBar();

window.addEventListener('scroll', () => {
  const scrollTop = window.scrollY;
  const docHeight = document.documentElement.scrollHeight - window.innerHeight;
  const percent = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
  progressBar.style.width = `${percent}%`;
});

// ===========================
// Hero entrance animations on load
// ===========================
window.addEventListener('load', () => {
  const heroH1 = document.querySelector('.hero h1');
  const heroP = document.querySelector('.hero p');
  if (heroH1) { heroH1.classList.add('slide-left'); heroH1.style.animationDelay = '0.2s'; }
  if (heroP) { heroP.classList.add('slide-right'); heroP.style.animationDelay = '0.4s'; }
});

// ===========================
// Section header glow on intersect
// ===========================
const sectionHeaders = document.querySelectorAll('.section h2');
const headerObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => { if (entry.isIntersecting) entry.target.classList.add('glow-animation'); });
}, { threshold: 0.5 });
sectionHeaders.forEach(header => headerObserver.observe(header));

// Mark todo items completed
try { /* update todo statuses */ } catch(e) {}


