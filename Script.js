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

// ===========================
// Sign In (DevAccount - edit credentials here)
// ===========================
const DevAccount = {
  username: 'dev',
  password: '78945612'
};

const SIGN_IN_SESSION_KEY = 'signin-signed-in';

const signInBtn = document.getElementById('signInBtn');
const signInLabel = document.getElementById('signInLabel');
const signInModal = document.getElementById('signIn-modal');
const signInDevBtn = document.getElementById('signIn-dev-btn');
const signInDevForm = document.getElementById('signIn-dev-form');
const signInUsername = document.getElementById('signIn-username');
const signInPassword = document.getElementById('signIn-password');
const signInError = document.getElementById('signIn-error');
const signInCancel = document.getElementById('signIn-cancel');
const signInSubmit = document.getElementById('signIn-submit');

function isSignedIn() {
  return sessionStorage.getItem(SIGN_IN_SESSION_KEY) === '1';
}

function updateSignInUI() {
  if (signInBtn && signInLabel) {
    const icon = signInBtn.querySelector('i');
    if (isSignedIn()) {
      signInLabel.textContent = 'Sign Out';
      if (icon) { icon.classList.remove('fa-right-to-bracket'); icon.classList.add('fa-right-from-bracket'); }
      signInBtn.title = 'Sign out';
    } else {
      signInLabel.textContent = 'Sign In';
      if (icon) { icon.classList.remove('fa-right-from-bracket'); icon.classList.add('fa-right-to-bracket'); }
      signInBtn.title = 'Sign in';
    }
  }
}

function openSignInModal() {
  if (signInModal) {
    signInModal.style.display = 'flex';
    if (signInDevForm) signInDevForm.style.display = 'none';
    if (signInUsername) signInUsername.value = '';
    if (signInPassword) signInPassword.value = '';
    if (signInError) { signInError.style.display = 'none'; signInError.textContent = ''; }
  }
}

function toggleDevForm() {
  if (signInDevForm) {
    const isHidden = signInDevForm.style.display === 'none' || !signInDevForm.style.display;
    signInDevForm.style.display = isHidden ? 'block' : 'none';
    if (isHidden && signInUsername) signInUsername.focus();
  }
}

function closeSignInModal() {
  if (signInModal) signInModal.style.display = 'none';
}

function handleSignInSubmit() {
  const user = (signInUsername?.value || '').trim();
  const pass = signInPassword?.value || '';
  if (user === DevAccount.username && pass === DevAccount.password) {
    sessionStorage.setItem(SIGN_IN_SESSION_KEY, '1');
    closeSignInModal();
    updateSignInUI();
    updateDevToolsVisibility();
    showToast(`Welcome, ${DevAccount.username}!`);
  } else {
    if (signInError) {
      signInError.textContent = 'Invalid username or password';
      signInError.style.display = 'block';
    }
  }
}

function handleSignInClick() {
  if (isSignedIn()) {
    sessionStorage.removeItem(SIGN_IN_SESSION_KEY);
    updateSignInUI();
    updateDevToolsVisibility();
    showToast('Signed out');
  } else {
    openSignInModal();
  }
}

if (signInBtn) signInBtn.addEventListener('click', handleSignInClick);

// ===========================
// Dev: Section reordering + Commit to GitHub
// ===========================
const SECTION_ORDER_KEY = 'dev-section-order';
let sortableInstance = null;

function updateDevToolsVisibility() {
  if (isSignedIn()) initSortable(); else destroySortable();
}

function saveSectionOrder() {
  const main = document.getElementById('main-sortable');
  if (!main) return;
  const ids = [...main.children].map(el => el.dataset.sectionId).filter(Boolean);
  localStorage.setItem(SECTION_ORDER_KEY, JSON.stringify(ids));
}

function restoreSectionOrder() {
  const main = document.getElementById('main-sortable');
  const saved = localStorage.getItem(SECTION_ORDER_KEY);
  if (!main || !saved) return;
  try {
    const order = JSON.parse(saved);
    const sections = [...main.children];
    const byId = {};
    sections.forEach(el => { const id = el.dataset?.sectionId; if (id) byId[id] = el; });
    order.forEach(id => { if (byId[id]) main.appendChild(byId[id]); });
  } catch (e) {}
}

function initSortable() {
  const main = document.getElementById('main-sortable');
  if (!main || sortableInstance || typeof Sortable === 'undefined') return;
  sortableInstance = Sortable.create(main, {
    animation: 200,
    ghostClass: 'sortable-ghost',
    chosenClass: 'sortable-chosen',
    onEnd: () => saveSectionOrder()
  });
}

function destroySortable() {
  if (sortableInstance) { sortableInstance.destroy(); sortableInstance = null; }
}

// Restore order on load, then update dev tools after sign-in UI
restoreSectionOrder();
if (signInDevBtn) signInDevBtn.addEventListener('click', toggleDevForm);
if (signInCancel) signInCancel.addEventListener('click', closeSignInModal);
if (signInSubmit) signInSubmit.addEventListener('click', handleSignInSubmit);
if (signInModal) signInModal.addEventListener('click', (e) => { if (e.target === signInModal) closeSignInModal(); });
if (signInPassword) signInPassword.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleSignInSubmit(); });
if (signInUsername) signInUsername.addEventListener('keypress', (e) => { if (e.key === 'Enter') signInPassword?.focus(); });

updateSignInUI();
updateDevToolsVisibility();

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
// Python ML Chatbot (TF-IDF + Cosine Similarity) with Learning
// ===========================
const chatToggle = document.getElementById('chat-toggle');
const chatWidget = document.getElementById('chat-widget');
const chatClose = document.getElementById('chat-close');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatMessages = document.getElementById('chat-messages');
const chatTeach = document.getElementById('chat-teach');
const learnModal = document.getElementById('learn-modal');
const learnPhrase = document.getElementById('learn-phrase');
const learnResponse = document.getElementById('learn-response');
const learnCancel = document.getElementById('learn-cancel');
const learnSave = document.getElementById('learn-save');
const learnedCountEl = document.getElementById('learned-count');
const mlStatusEl = document.getElementById('ml-status');

let pyodide = null;
let mlChatReady = false;

// Database: Load learned responses from localStorage
function loadLearnedData() {
  try {
    const data = localStorage.getItem('ml-chatbot-learned');
    return data ? JSON.parse(data) : {};
  } catch (e) { return {}; }
}

// Database: Save learned responses to localStorage
function saveLearnedData(data) {
  try {
    localStorage.setItem('ml-chatbot-learned', JSON.stringify(data));
  } catch (e) { console.error('Failed to save learned data:', e); }
}

// Update learned count display
function updateLearnedCount() {
  const data = loadLearnedData();
  const count = Object.keys(data).length;
  if (learnedCountEl) learnedCountEl.textContent = count;
}

// Update ML status display (Llama AI connection or ML fallback)
function updateMLStatus(status) {
  if (mlStatusEl) mlStatusEl.textContent = status;
}

// Initialize Python ML Chatbot with Learning
async function initMLChatbot() {
  updateMLStatus('Loading...');
  try {
    pyodide = await loadPyodide({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.23.4/full/",
    });

    const pythonCode = `
import re
import math
from collections import Counter, defaultdict

class TFIDF:
    def __init__(self):
        self.documents = []
        self.vocabulary = set()
        self.idf = {}
        self.doc_vectors = []
    
    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r'\\b[a-z]+\\b', text)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'can', 'to',
                     'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
                     'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'between', 'under', 'again', 'further', 'then',
                     'once', 'here', 'there', 'when', 'where', 'why', 'how',
                     'all', 'each', 'few', 'more', 'most', 'other', 'some',
                     'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                     'because', 'until', 'while', 'this', 'that', 'these',
                     'those', 'am', 'it', 'its', 'i', 'me', 'my', 'myself',
                     'we', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him',
                     'his', 'she', 'her', 'hers', 'they', 'them', 'their',
                     'what', 'which', 'who', 'whom'}
        return [t for t in tokens if t not in stopwords and len(t) > 1]
    
    def fit(self, documents):
        self.documents = documents
        doc_count = len(documents)
        term_doc_freq = defaultdict(int)
        
        for doc in documents:
            tokens = set(self.tokenize(doc))
            self.vocabulary.update(tokens)
            for token in tokens:
                term_doc_freq[token] += 1
        
        for term in self.vocabulary:
            self.idf[term] = math.log(doc_count / (term_doc_freq[term] + 1)) + 1
        
        self.doc_vectors = [self._compute_tfidf(doc) for doc in documents]
    
    def _compute_tfidf(self, text):
        tokens = self.tokenize(text)
        if not tokens:
            return {}
        tf = Counter(tokens)
        max_tf = max(tf.values()) if tf else 1
        vector = {}
        for term, count in tf.items():
            if term in self.idf:
                vector[term] = (count / max_tf) * self.idf[term]
        return vector
    
    def cosine_similarity(self, vec1, vec2):
        if not vec1 or not vec2:
            return 0.0
        common_terms = set(vec1.keys()) & set(vec2.keys())
        if not common_terms:
            return 0.0
        dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)
    
    def find_most_similar(self, query, threshold=0.1):
        query_vector = self._compute_tfidf(query)
        best_idx = -1
        best_score = threshold
        for idx, doc_vector in enumerate(self.doc_vectors):
            score = self.cosine_similarity(query_vector, doc_vector)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx, best_score

class MLChatbot:
    def __init__(self):
        self.learned_responses = {}  # Database for learned responses
        self.ml_knowledge = [
            ("hello hi hey greetings hola welcome", [
                "Hello! \ud83d\udc4b I'm Imrane's ML-powered chatbot using TF-IDF!",
                "Hi there! \ud83d\ude0a I'm Imrane's assistant. Ask about projects, skills, or games!",
                "Hey! Welcome to Imrane's portfolio! I use cosine similarity to understand you!"
            ]),
            ("project projects game games build built code portfolio work", [
                "\ud83c\udfae I have built 12+ games: Snake, 2048, RPS AI, Memory Match, ML games!",
                "\ud83d\udce6 My portfolio: vanilla JS games, Canvas API, Godot Engine, Python AI!",
                "\ud83c\udfaf Check the Games section for playable demos with AI opponents!"
            ]),
            ("skill skills technology tech language languages python javascript", [
                "\ud83d\udcbb Skills: JavaScript, Python, HTML/CSS, Canvas API, Godot, ML/AI!",
                "\ud83d\udee0\ufe0f Tech stack: Vanilla JS, Python (ML), Godot GDScript, WebAssembly!",
                "\u2699\ufe0f I work with: Neural Networks, TF-IDF, Pattern Recognition, NLP!"
            ]),
            ("ai artificial intelligence machine learning ml neural network algorithm", [
                "\ud83e\udd16 AI/ML is my passion! I built neural networks, digit classifiers!",
                "\ud83e\udde0 This chat uses TF-IDF + Cosine Similarity - real ML!",
                "\ud83c\udf93 I implement: TF-IDF vectorization, cosine similarity, backpropagation!"
            ]),
            ("contact email reach message connect", [
                "\ud83d\udce7 Email: imrane2015su@gmail.com - Let's connect!",
                "\ud83d\udc8c Reach me at: imrane2015su@gmail.com for collaborations!",
                "\ud83e\udd1d Contact: imrane2015su@gmail.com - Always happy to chat!"
            ]),
            ("chatbot bot how work algorithm yourself about explain tfidf", [
                "\ud83e\udd16 I use TF-IDF to convert text to vectors!",
                "\ud83e\udde0 My algorithm: tokenize \u2192 remove stopwords \u2192 TF-IDF \u2192 cosine similarity!",
                "\ud83d\udcca I find the most similar training phrase and return its response!"
            ]),
            ("time duration long much weeks days timeline", [
                "\u23f1\ufe0f Project times: Simple games 1 week, complex 2-3 weeks, large 3-4 weeks.",
                "\ud83d\udcc5 Most projects take 2-3 weeks depending on complexity."
            ])
        ]
        self.tfidf = TFIDF()
        self._train_tfidf()
    
    def _train_tfidf(self):
        training_docs = [phrase for phrase, _ in self.ml_knowledge]
        # Add learned responses to training
        for phrase in self.learned_responses.keys():
            training_docs.append(phrase)
        self.tfidf.fit(training_docs)
    
    def learn(self, phrase, response):
        """Learn a new phrase-response pair and save to database"""
        processed = phrase.lower().strip()
        if processed not in self.learned_responses:
            self.learned_responses[processed] = []
        self.learned_responses[processed].append(response)
        self._train_tfidf()  # Retrain with new data
        return len(self.learned_responses)
    
    def load_learned(self, data):
        """Load learned responses from external storage"""
        self.learned_responses = data
        self._train_tfidf()
    
    def chat(self, user_input):
        processed = user_input.lower().strip()
        best_idx, confidence = self.tfidf.find_most_similar(processed, threshold=0.15)
        
        if best_idx >= 0:
            num_base = len(self.ml_knowledge)
            if best_idx >= num_base:
                # It's a learned response
                learned_idx = best_idx - num_base
                learned_phrase = list(self.learned_responses.keys())[learned_idx]
                responses = self.learned_responses[learned_phrase]
            else:
                # Base knowledge
                _, responses = self.ml_knowledge[best_idx]
            import random
            response = random.choice(responses)
            return response, confidence
        
        fallbacks = [
            "\ud83e\udd14 Interesting! Try asking about projects, skills, or games.",
            "\ud83d\udcad Ask me about AI/ML, games, or contact info!",
            "\u2728 Try keywords like 'skills', 'games', 'AI', or 'contact'!",
            "\ud83c\udf93 Click the teach button (\ud83c\udf93) to help me learn new things!"
        ]
        import random
        return random.choice(fallbacks), 0.0

chatbot = MLChatbot()
`;

    await pyodide.runPythonAsync(pythonCode);
    
    // Load learned data from localStorage and inject into Python
    const learnedData = loadLearnedData();
    if (Object.keys(learnedData).length > 0) {
      const dataJson = JSON.stringify(learnedData);
      await pyodide.runPythonAsync(`
import json
learned_data = json.loads('${dataJson.replace(/'/g, "\\'")}')
chatbot.load_learned(learned_data)
`);
    }
    
    mlChatReady = true;
    if (!checkLlamaConnection()) updateMLStatus('ML Ready ✓');
    updateLearnedCount();
    console.log('\ud83e\udde0 ML Chatbot loaded successfully!');
  } catch (error) {
    console.error('ML Chatbot failed to load:', error);
    mlChatReady = false;
    updateMLStatus('Error');
  }
}

// Teach the bot a new response
async function teachBot(phrase, response) {
  if (!mlChatReady || !pyodide) {
    showToast('ML not ready yet!');
    return false;
  }
  
  try {
    // Teach Python chatbot
    await pyodide.runPythonAsync(`
chatbot.learn("""${phrase.replace(/"/g, '\\"')}""", """${response.replace(/"/g, '\\"')}""")
`);
    
    // Save to localStorage database
    const data = loadLearnedData();
    const key = phrase.toLowerCase().trim();
    if (!data[key]) data[key] = [];
    data[key].push(response);
    saveLearnedData(data);
    
    updateLearnedCount();
    return true;
  } catch (error) {
    console.error('Learn error:', error);
    return false;
  }
}

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

function renderMessage(role, text, confidence = null) {
  const wrapper = document.createElement('div');
  wrapper.className = `chat-message ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = text;
  if (role === 'bot' && confidence !== null && confidence > 0) {
    const confSpan = document.createElement('div');
    confSpan.style.cssText = 'font-size:10px;color:#888;margin-top:4px;';
    confSpan.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
    bubble.appendChild(confSpan);
  }
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Fallback for when ML is not ready
function getFallbackResponse(text) {
  const lower = text.toLowerCase();
  const responses = {
    'project|game|build': "\ud83c\udfae I have 12+ games! Check the Games section.",
    'skill|tech|python|javascript': "\ud83d\udcbb Skills: JS, Python, Godot, ML, Canvas API!",
    'ai|machine learning|ml': "\ud83e\udd16 AI/ML is my passion! TF-IDF, neural networks!",
    'contact|email': "\ud83d\udce7 Email: imrane2015su@gmail.com",
    'hello|hi|hey': "Hello! \ud83d\udc4b Ask me about my portfolio!"
  };
  for (const [keywords, reply] of Object.entries(responses)) {
    for (const kw of keywords.split('|')) {
      if (lower.includes(kw)) return reply;
    }
  }
  return "\u2728 Ask about my projects, skills, or games!";
}

// Free AI Agent - Puter.js (Llama 3.3 70B) - No API key required
const FREE_AI_SYSTEM_PROMPT = `You are a friendly AI assistant for Imrane Naitembark's portfolio. You know about: his 12+ games (Snake, 2048, Memory Match, Fragime, Hamood, ML Digit Classifier, etc.), skills (JavaScript, Python, Godot, AI/ML, TF-IDF), and contact: imrane2015su@gmail.com. Keep replies concise (under 100 words), helpful, and use emojis occasionally. Focus on portfolio topics.`;

const LLAMA_MODEL = 'meta-llama/llama-3.3-70b-instruct';

function isPuterReady() {
  return typeof puter !== 'undefined' && puter?.ai?.chat;
}

async function getFreeAIResponse(text) {
  if (!isPuterReady()) return null;
  try {
    const response = await puter.ai.chat(text, {
      model: LLAMA_MODEL,
      systemPrompt: FREE_AI_SYSTEM_PROMPT
    });
    const content = response?.message?.content || response?.content || (typeof response === 'string' ? response : null);
    return content ? { response: content.trim(), confidence: 1, freeAI: true } : null;
  } catch (e) {
    return null;
  }
}

// Check Puter/Llama connection and update status
function checkLlamaConnection() {
  if (isPuterReady()) {
    updateMLStatus('Llama Ready ✓');
    return true;
  }
  return false;
}

// Get ML response from Python
async function getMLResponse(text) {
  if (!mlChatReady || !pyodide) {
    return { response: getFallbackResponse(text), confidence: 0 };
  }
  
  try {
    const result = await pyodide.runPythonAsync(`
import json
result = chatbot.chat("""${text.replace(/"/g, '\\"')}""")
json.dumps({'response': result[0], 'confidence': float(result[1])})
`);
    return JSON.parse(result);
  } catch (error) {
    console.error('ML response error:', error);
    return { response: getFallbackResponse(text), confidence: 0 };
  }
}

// Initialize chat UI and history
let chatHistory = loadChatHistory();
if (chatHistory.length) {
  chatHistory.forEach(item => renderMessage(item.role, item.text, item.confidence));
}

function openChat() {
  chatWidget.setAttribute('aria-hidden', 'false');
  chatWidget.style.display = 'flex';
  chatInput.focus();
  chatToggle.classList.add('active');
  sessionStorage.setItem('chat-open', '1');
  
  // Initialize ML chatbot on first open (fallback)
  if (!pyodide && typeof loadPyodide !== 'undefined') {
    initMLChatbot();
  }
  // Check Puter/Llama connection - retry every 500ms for 5s
  let attempts = 0;
  const checkInterval = setInterval(() => {
    if (checkLlamaConnection() || ++attempts >= 10) clearInterval(checkInterval);
  }, 500);
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

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = (chatInput.value || '').trim();
  if (!text) { showToast('Type a message'); return; }

  // Render user message
  renderMessage('user', text);
  chatHistory.push({ role: 'user', text });
  saveChatHistory(chatHistory);
  chatInput.value = '';

  // Show typing indicator
  const typingDiv = document.createElement('div');
  typingDiv.className = 'chat-message bot';
  typingDiv.innerHTML = '<div class="bubble">\ud83e\udde0 Thinking...</div>';
  chatMessages.appendChild(typingDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Get response: try Free AI (Puter/Llama) first, fallback to ML
  const delay = prefersReducedMotion ? 0 : 300;
  setTimeout(async () => {
    let result = await getFreeAIResponse(text);
    if (!result) result = await getMLResponse(text);
    const { response, confidence } = result;
    typingDiv.remove();
    renderMessage('bot', response, confidence);
    if (result.freeAI) {
      const lastBubble = chatMessages.lastElementChild?.querySelector('.bubble');
      if (lastBubble) {
        const badge = document.createElement('div');
        badge.style.cssText = 'font-size:9px;color:#4ade80;margin-top:2px;';
        badge.textContent = '🦙 Free AI (Llama)';
        lastBubble.appendChild(badge);
      }
    }
    chatHistory.push({ role: 'bot', text: response, confidence });
    saveChatHistory(chatHistory);
  }, delay);
});

// Simple keyboard: Esc to close chat or modal
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    if (learnModal && learnModal.style.display === 'flex') {
      closeLearnModal();
    } else if (chatWidget.style.display === 'flex') {
      closeChat();
    }
  }
});

// ===========================
// Learning Modal Functions
// ===========================
function openLearnModal() {
  if (learnModal) {
    learnModal.style.display = 'flex';
    if (learnPhrase) learnPhrase.focus();
  }
}

function closeLearnModal() {
  if (learnModal) {
    learnModal.style.display = 'none';
    if (learnPhrase) learnPhrase.value = '';
    if (learnResponse) learnResponse.value = '';
  }
}

async function saveLearnedResponse() {
  const phrase = learnPhrase ? learnPhrase.value.trim() : '';
  const response = learnResponse ? learnResponse.value.trim() : '';
  
  if (!phrase || !response) {
    showToast('Please fill in both fields!');
    return;
  }
  
  const success = await teachBot(phrase, response);
  
  if (success) {
    closeLearnModal();
    renderMessage('bot', `✅ Learned! I'll remember "${phrase.substring(0, 25)}..." now!`, 1.0);
    showToast('Saved to database!');
  } else {
    showToast('Failed to save. Try again!');
  }
}

// Event listeners for teach button and modal
if (chatTeach) {
  chatTeach.addEventListener('click', openLearnModal);
}

if (learnCancel) {
  learnCancel.addEventListener('click', closeLearnModal);
}

if (learnSave) {
  learnSave.addEventListener('click', saveLearnedResponse);
}

// Close modal when clicking outside
if (learnModal) {
  learnModal.addEventListener('click', (e) => {
    if (e.target === learnModal) closeLearnModal();
  });
}

// Initialize learned count on page load
updateLearnedCount();

// ===========================
// Plugins: AOS (Animate On Scroll) + GLightbox (Image Lightbox)
// ===========================
window.addEventListener('DOMContentLoaded', () => {
  if (typeof AOS !== 'undefined' && !prefersReducedMotion) {
    AOS.init({ duration: 600, offset: 80, once: true });
  }
  if (typeof GLightbox !== 'undefined') {
    GLightbox({ selector: '.glightbox', touchNavigation: true });
  }
});

