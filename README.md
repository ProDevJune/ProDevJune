# 🧠 ProDevJune GitHub Pages Blog

Welcome to the **ProDevJune Blog**, a GitHub Pages-powered blog for programming and AI enthusiasts. This repository is built using the **Chirpy Jekyll Theme**, enhanced with features from **Minimal Mistakes** for advanced tagging and categorization.

---

## 🔍 Overview
- **Main URL:** [https://prodevjune.github.io/ProDevJune](https://prodevjune.github.io/ProDevJune)
- **Base URL:** `/ProDevJune`
- **Blog Stack:** Jekyll + Chirpy Theme + Minimal Mistakes Tag/Category Archives
- **Primary Topics:** AI, Programming, DevTips, Git, Tools, Project Docs

---

## 🚀 Features

### ✅ Blog Core
- Markdown-based posts in `_posts/`
- SEO-friendly structure
- Category and tag support
- Clean responsive design (Chirpy Theme)

### ✅ Extended Features
- 🏷️ **Tag & Category Pages**: Based on Minimal Mistakes-style archives
- 🧮 **Math & Diagrams**: LaTeX (MathJax) and Mermaid support enabled
- 🛠️ **Post Generator**: `new-post.sh` automates post creation
- 💅 **Code Quality Tools**: Prettier, ESLint, Stylelint, Husky pre-commit
- 🌐 **GitHub Actions**: Auto-deploys on push to `main`

---

## 🛠 Local Development Guide

### 1. Clone and Setup
```bash
git clone https://github.com/ProDevJune/ProDevJune.git
cd ProDevJune
pnpm install
```

### 2. Serve Locally
```bash
pnpm exec jekyll serve
```
Visit: [http://localhost:4000/ProDevJune/](http://localhost:4000/ProDevJune/)

---

## 📝 Writing a Post

### Use Script:
```bash
bash new-post.sh your-post-title
```
> Generates `_posts/YYYY-MM-DD-your-post-title.md`

### Add Front Matter Example:
```yaml
---
title: "AI 리뷰 분석기"
date: 2025-03-27 12:00:00 +0900
tags: [AI, 리뷰, NLP]
categories: [프로젝트]
---
```

### Tag Archive Enabled
- `/tag/` shows all tags
- `/tag/ai/`, `/category/프로젝트/` available via auto-links

---

## 📂 Directory Overview
```
ProDevJune/
├── _posts/            # Markdown posts
├── _tabs/             # About, Projects, etc.
├── _data/navigation.yml  # Menu links
├── tag/, category/    # Tag & category pages
├── _includes/         # tag-list.html, category-list.html
├── .husky/            # Pre-commit hook (husky)
├── .vscode/           # Autoformatting config
├── tools/             # run.sh, test.sh
├── new-post.sh        # Post generation script
└── ...
```

---

## 🔧 Dev Tools Configured

- `.editorconfig` – whitespace, line endings
- `.prettierrc`, `.stylelintrc.json`, `.eslintrc.json`
- `.husky/pre-commit` – lint-staged auto-run
- `package.json` includes scripts and dev dependencies

---

## 🔐 Deployment (GitHub Actions)
- Uses `.github/workflows/pages-deploy.yml`
- Triggered on push to `main`
- Publishes to GitHub Pages: `https://prodevjune.github.io/ProDevJune`

---

## 💬 Contact
- Maintainer: **ProDevJune**
- Email: gd2u37@gmail.com

---

## 📄 License
This repository is **not open source**. All rights reserved. Redistribution or reuse of the code without permission is prohibited.

