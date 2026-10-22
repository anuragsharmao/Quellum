# Quellum — A Hallucination Aware RAG

A pixel-faithful, pure **Vanilla JavaScript / HTML5 / CSS3** recreation of the Quellum homepage design reference.

**Zero dependencies. No React. No build steps required.**

---

## 🎨 Visual Design Specification Recreated

1. **Atmospheric Luminous Background:**
   - Soft cream-white base tone (`#faf8f5`).
   - Diffuse pastel edge glows: sky blue (top-left), peach & lavender (top-right), rose pink & soft peach (bottom-left), pale azure (bottom-right).
   - Translucent flowing gradient wave ribbons in the margins with a central luminous wash for readability.
2. **Top Navigation:**
   - **Corrected Q Logo:** Uses the **pink-to-orange/yellow gradient** (`#ec4899` &rarr; `#f43f5e` &rarr; `#f97316` &rarr; `#fbbf24`) matching the "llum" segment of the wordmark, with circular body and distinct tail.
   - Clean floating navigation without container boxes or solid strips.
   - Centered tabs with refined dark charcoal typography (`#0f172a`), crisp outline icons, and cyan-to-purple gradient underline on `Home`.
   - Top-right outline theme toggle button.
3. **Hero Section:**
   - Full-spectrum gradient Quellum wordmark (`#00d2ff` &rarr; `#6366f1` &rarr; `#ec4899` &rarr; `#f97316` &rarr; `#fbbf24`).
   - Delicate flowing decorative curves flanking the wordmark.
   - Authoritative dark navy heading: **“A Hallucination Aware RAG”** (`#0f172a`).
   - Supporting text: *“Upload a document or paste text to check how well an answer is supported by the available evidence.”*
4. **Primary Buttons & Action Elements:**
   - **Black / Charcoal Primary Action Buttons (`#111827`):** “Choose PDF” and “Analyze Text” buttons use solid dark charcoal styling with crisp white typography, subtle borders, and smooth tactile elevation.
   - Refined segmented control with dark charcoal active label and black outline icons.
   - Subtle dark charcoal outline icons inside the pastel badge containers in the bottom feature row.
5. **Handwritten Annotations:**
   - Upper-right:
     > *“Better evidence.*  
     > *More reliable answers.”*  
     Styled in natural handwritten script (`Caveat`), royal ink blue (`#1d4ed8`), tilted -5°, with a pink-to-orange curved flourish underline.
   - Lower-left:
     > *“Ground*  
     > *every answer.”*  
     Handwritten script, tilted -8°, with a pink-to-orange curved underline.
6. **Floating Illustrations:**
   - **Upper-Left:** Overlapping documents with red "PDF" tag, yellow highlighted passage lines, floating "Key findings" (cyan sparkle) and "Tables & charts" (pink table) pills, and curved ink arrow.
   - **Upper-Right:** Floating "Analyzing answer..." card with highlighted sentence bar, "Potential hallucination" alert badge, connected "Source / Page 4" pill, and curved arrow.
   - **Lower-Left:** "Source context" preview card with cyan highlighted lines.
   - **Lower-Right:** Overlapping evidence pages with yellow highlighted lines and 3D vector blue magnifying glass with specular lens reflection and spark rays.
   - Slow, independent floating keyframe animations respecting `prefers-reduced-motion`.
7. **Main Upload and Paste-Text Panel:**
   - Rounded white panel with subtle border and layered soft shadows.
   - Segmented switch (`Upload PDF` / `Paste Text`) with cyan-to-purple gradient active indicator.
   - Dual-layout body with central vertical divider and *"or"* circular badge:
     - **Upload PDF Mode:** Dashed drop zone, outline cloud upload icon, “Upload a PDF”, charcoal button “Choose PDF”, “PDF only · Max file size 20 MB”, drag-and-drop file detection, and selected file chip with remove button.
     - **Paste Text Mode:** Purple document icon, “Paste Text”, rounded textarea, word/character counter, quick sample text filler, and charcoal button “Analyze Text”.
   - Interactive evidence inspection modal showing grounding score, grounded citations, and flagged hallucination warnings.
8. **Bottom Feature Row:**
   - Four columns with vertical dividers:
     1. **Hallucination Detection** (Shield, soft blue badge)
     2. **Evidence Grounding** (Document, soft lilac badge)
     3. **Sentence-Level Review** (Magnifying glass, soft amber badge)
     4. **Clear Findings** (Target / Bullseye, soft mint badge)

---

## 🚀 How to Run

### Option 1: Direct File Open (Zero Setup)
Simply double-click or open [`frontend/index.html`](file:///c:/Users/anura/Documents/GitHub/Quellum/frontend/index.html) in Chrome, Edge, Safari, or Firefox.

### Option 2: Local Web Server
```bash
# Using Python
python -m http.server 3000 --directory frontend

# Or using npm
npm start
```
Then open `http://localhost:3000` in your browser.

---

## 📁 File Structure

```text
Quellum/
├── frontend/
│   ├── index.html     # Semantic HTML5 markup with embedded SVGs
│   ├── style.css      # Modern vanilla CSS3 layout, gradients, and animations
│   ├── script.js      # Pure Vanilla JS handling drag-drop, tabs, and modals
│   └── package.json   # Optional runner script
├── package.json       # Root runner script
└── README.md          # Project documentation
```
