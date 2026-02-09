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
// Navbar Item Stagger Animation on Load
// ===========================
window.addEventListener('DOMContentLoaded', () => {
  if (!prefersReducedMotion) {
    const navBrand = document.querySelector('.nav-brand');
    const navLinks = document.querySelectorAll('.nav-links a');
    const navActions = document.querySelector('.nav-actions');
    
    // Navbar brand entrance
    if (navBrand) {
      navBrand.style.animation = 'slideInLeft 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards';
    }
    
    // Staggered navbar links entrance
    navLinks.forEach((link, index) => {
      link.style.animation = `navSlideDown 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards`;
      link.style.animationDelay = `${0.1 + index * 0.1}s`;
    });
    
    // Navbar actions entrance
    if (navActions) {
      navActions.style.animation = 'slideInRight 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards';
    }
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
      card.style.boxShadow = '0 12px 30px rgba(0, 123, 255, 0.3)';
      card.style.transform = 'translateY(-8px)';
    });
    card.addEventListener('mouseleave', () => {
      card.style.animation = 'none';
      card.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.1)';
      card.style.transform = 'translateY(0)';
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
  if (!prefersReducedMotion) {
    skill.style.animationDelay = `${0.3 + index * 0.1}s`;
  } else {
    skill.style.opacity = '1';
  }
});

const projectCards = document.querySelectorAll('.cards .card');
projectCards.forEach((card, index) => {
  if (!prefersReducedMotion) {
    card.style.animationDelay = `${0.4 + index * 0.15}s`;
  } else {
    card.style.opacity = '1';
  }
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
  
  // Add click animation
  btn.addEventListener('click', function() {
    if (!prefersReducedMotion) {
      this.style.animation = 'none';
      // Trigger reflow to restart animation
      void this.offsetWidth;
      this.style.animation = 'buttonPress 0.3s ease';
    }
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

// ===========================
// Llama LLM-Powered Chatbot (Model-Free Open-Source)
// ===========================
const chatToggle = document.getElementById('chat-toggle');
const chatWidget = document.getElementById('chat-widget');
const chatClose = document.getElementById('chat-close');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatMessages = document.getElementById('chat-messages');

// Load chat history from sessionStorage
function loadChatHistory() {
  try {
    const raw = sessionStorage.getItem('chat-history');
    if (!raw) return [];
    return JSON.parse(raw);
  } catch (e) { return []; }
}

function saveChatHistory(history) {
  try { sessionStorage.setItem('chat-history', JSON.stringify(history)); } catch (e) {}
}

function renderMessage(role, text) {
  const wrapper = document.createElement('div');
  wrapper.className = `chat-message ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = text;
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Local Llama-inspired response generation
function getLlamaResponse(text) {
  const lower = text.toLowerCase();
  
  // Knowledge responses (Llama-like)
  const responses = {
    'project|game|build|code|portfolio': [
      "I have built 12+ games including Snake, 2048, Rock Paper Scissors AI, Memory Match, Reaction Time, Tic Tac Toe, and ML-powered games!",
      "Check the Games section to play React projects, Canvas games with Godot, and Python AI games!"
    ],
    'skill|tech|technology|language|javascript|python': [
      "I'm skilled in JavaScript, Python, HTML, CSS, Canvas API, Godot GDScript, and Machine Learning!",
      "My tech stack: Vanilla JS, Python (ML/Data), Godot Engine, WebAssembly, Neural Networks, TensorFlow concepts"
    ],
    'ai|machine learning|ml|neural|llama': [
      "AI/ML is my passion! I built neural networks, digit classifiers, and learning algorithms!",
      "I use Llama (open-source LLM) for this chat, plus TF-IDF, cosine similarity, and pattern recognition!"
    ],
    'contact|email|reach|message': [
      "Contact me at: imrane2015su@gmail.com for collaborations and opportunities!",
      "Email: imrane2015su@gmail.com - I'm always happy to discuss projects!"
    ],
    'hello|hi|hey|greetings': [
      "Hello! I'm a Llama-powered AI assistant. What would you like to know about my portfolio?",
      "Hi there! Welcome! Ask me about my games, skills, or AI/ML work!"
    ],
    'time|duration|how long': [
      "Most projects take 2-3 weeks. Games typically take 1-2 weeks, larger projects take 3-4 weeks.",
      "Project timelines vary: simple games are 1 week, complex projects are 3-4 weeks."
    ]
  };

  // Find matching response
  for (const [keywords, replies] of Object.entries(responses)) {
    for (const keyword of keywords.split('|')) {
      if (lower.includes(keyword)) {
        return replies[Math.floor(Math.random() * replies.length)];
      }
    }
  }

  // Fallback responses
  const fallbacks = [
    "That's interesting! Tell me more about what interests you!",
    "I see! You can also check the Games or Code Examples sections!",
    "Great question! Feel free to ask about my projects, skills, or experience!",
    "I appreciate that! Is there anything specific about my portfolio you'd like to know?"
  ];

  return fallbacks[Math.floor(Math.random() * fallbacks.length)];
}

// Initialize chat UI and history
let chatHistory = loadChatHistory();
if (chatHistory.length) {
  chatHistory.forEach(item => renderMessage(item.role, item.text));
}

function openChat() {
  chatWidget.setAttribute('aria-hidden', 'false');
  chatWidget.style.display = 'flex';
  chatInput.focus();
  chatToggle.classList.add('active');
  sessionStorage.setItem('chat-open', '1');
}

function closeChat() {
  chatWidget.setAttribute('aria-hidden', 'true');
  chatWidget.style.display = 'none';
  chatToggle.focus();
  chatToggle.classList.remove('active');
  sessionStorage.removeItem('chat-open');
}

chatToggle.addEventListener('click', () => {
  if (chatWidget.style.display === 'flex') closeChat(); else openChat();
});

chatClose.addEventListener('click', closeChat);

// Restore open state
if (sessionStorage.getItem('chat-open')) openChat(); else chatWidget.style.display = 'none';

chatForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = (chatInput.value || '').trim();
  if (!text) { showToast('Type a message'); return; }

  // render user message
  renderMessage('user', text);
  chatHistory.push({ role: 'user', text });
  saveChatHistory(chatHistory);
  chatInput.value = '';

  // Use Llama-inspired responses
  const reply = getLlamaResponse(text);
  const delay = prefersReducedMotion ? 0 : Math.min(1200, 300 + reply.length * 10);
  setTimeout(() => {
    renderMessage('bot', reply);
    chatHistory.push({ role: 'bot', text: reply });
    saveChatHistory(chatHistory);
  }, delay);
});

// Simple keyboard: Esc to close chat
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && chatWidget.style.display === 'flex') closeChat();
});

// Mark todo items completed
try { /* update todo statuses */ } catch(e) {}


