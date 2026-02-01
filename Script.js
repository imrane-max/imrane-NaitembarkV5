// Dark Mode Toggle
const themeBtn = document.getElementById("ThemBtn");
const ViweBtn = document.getElementById

themeBtn.addEventListener("click", () => {
  document.body.classList.toggle("dark-mode");
  themeBtn.textContent = document.body.classList.contains("dark-mode") ? "Light" : "Dark";
});

// Fade-in Animation
const fadeElements = document.querySelectorAll('.fade-in');
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if(entry.isIntersecting) entry.target.classList.add('visible');
  });
}, { threshold: 0.2 });

fadeElements.forEach(el => observer.observe(el));



