// Dark Mode Toggle
const themeBtn = document.getElementById("ThemBtn");
const textEmail = "imrane2015su@gmial.com";
const DiscordUser = "imax__max";

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


function copyEmail() {
  navigator.clipboard.writeText(textEmail);
  alert("Copy Email");
};

function copyDiscord() {
  navigator.clipboard.writeText(DiscordUser);
  alert("Copy Discord User");
};
