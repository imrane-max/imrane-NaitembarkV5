# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Personal portfolio website for Imrane Naitembark showcasing web development, game development, and AI/ML projects. Static site with no build tools—just open `index.html` in a browser.

## Tech Stack

- **Frontend**: HTML5, CSS3, Vanilla JavaScript (ES6+)
- **Games**: Canvas API, self-contained HTML files with inline JS
- **AI/ML demos**: Python via Pyodide (WebAssembly), TF-IDF, Neural Networks
- **Game Engine**: Godot GDScript examples
- **No frameworks or dependencies** - pure vanilla code

## Development Commands

```powershell
# Serve locally (Python)
python -m http.server 8000

# Or use VS Code Live Server extension

# Run standalone chatbot
python chatbot.py
```

No build step, linting, or testing infrastructure exists. Changes are made directly to files and viewed in browser.

## Architecture

### Core Files
- `index.html` - Single-page portfolio with all sections
- `style.css` - Global styles, animations, dark mode (`.dark-mode` class)
- `Script.js` - Theme toggle, animations, chat widget, IntersectionObserver fades

### Key Patterns

**Theme Persistence**: Dark mode saved to `localStorage.getItem('site-theme')`, toggled via `body.dark-mode` class.

**Animations**: Use `.fade-in` class with IntersectionObserver for scroll reveals. Respects `prefers-reduced-motion`.

**Chat Widget**: ML-powered Python chatbot in `Script.js` using TF-IDF + Cosine Similarity via Pyodide. Supports learning and saves to `localStorage`.

### Folder Structure
- `games/` - Self-contained mini-games (each `.html` has all CSS/JS inline)
- `demos/` - Code examples (enemy AI demo with GDScript reference)
- Root HTML files (`ml-chat.html`, `python-chatbot.html`, etc.) - AI chatbot interfaces using Pyodide
- `Llaga-chat.html` - Real Llama 3.3 70B AI chatbot via Puter.js (FREE, no API key required)

### Games Architecture
Each game in `games/` follows this pattern:
```javascript
// 1. State initialization
let gameState = { score: 0, gameActive: true };

// 2. Render function (DOM or Canvas)
// 3. Update/game logic
// 4. Event listeners for input
// 5. Game loop (requestAnimationFrame for Canvas games)
```

Games use `localStorage` for high scores.

## Code Conventions

- Vanilla JS only—no jQuery, React, or other frameworks
- CSS animations via `@keyframes` defined in `style.css`
- Font: "Playwrite NZ Basic" (Google Fonts)
- Color scheme: `#007bff` (primary blue), `#4aa3ff` (dark mode blue)
- All games are single-file HTML with inline `<style>` and `<script>`

## Adding New Content

**New game**: Create `games/newgame.html` with inline CSS/JS, add card to Games section in `index.html`.

**New section**: Add `<section id="name" class="section fade-in">` in `index.html`, add nav link.

**New skill icon**: Add image to root, reference in Skills section with `<img>` inside `<span>`.
