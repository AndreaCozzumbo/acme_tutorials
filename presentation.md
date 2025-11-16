# GCN ACME Presentation

Click the button below to view the presentation:

<div style="text-align: center; margin: 3em 0;">
  <a href="slides/GCN_acme/index.html" target="_blank" style="display: inline-block; padding: 1.5em 3em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 12px; font-size: 1.3em; font-weight: 600; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4); transition: all 0.3s;">
    🎬 Open Presentation
  </a>
</div>

```{admonition} Presentation Options
:class: tip

- Click the button above to open the presentation in a new tab
- Use arrow keys (← →) to navigate between slides
- Press F for fullscreen mode
- Press ESC to exit fullscreen
```

```{note}
If you exported your Keynote presentation as HTML, place the files in the `slides/` directory.
Alternatively, you can embed a PDF or use services like Google Slides or Speaker Deck.
```

## Alternative Options

### Option 1: PDF Viewer
If you have a PDF version of your presentation:

```html
<iframe src="slides/presentation.pdf" 
        width="100%" 
        height="600px" 
        style="border: 1px solid #e5e7eb;">
</iframe>
```

### Option 2: Google Slides
For Google Slides presentations:

```html
<iframe src="https://docs.google.com/presentation/d/YOUR_PRESENTATION_ID/embed" 
        width="100%" 
        height="569px" 
        frameborder="0" 
        allowfullscreen="true" 
        mozallowfullscreen="true" 
        webkitallowfullscreen="true">
</iframe>
```

### Option 3: Speaker Deck
For Speaker Deck presentations:

```html
<iframe src="https://speakerdeck.com/player/YOUR_PRESENTATION_ID" 
        width="100%" 
        height="500px" 
        frameborder="0" 
        allowfullscreen>
</iframe>
```

## Instructions

To add your Keynote presentation:

1. **Export from Keynote**: File → Export To → HTML
2. **Create slides directory**: `mkdir slides` in the project root
3. **Copy exported files**: Place all HTML files in the `slides/` directory
4. **Update the iframe**: Change `src="slides/presentation.html"` to match your filename
5. **Rebuild the book**: Run `jupyter-book build .`

```{tip}
For best results, use Keynote's "Self-playing" HTML export option with navigation controls enabled.
```
