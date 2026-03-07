# Dave Zhou — Portfolio

Personal portfolio site. Template forked from [Studorlio](https://github.com/helfi92/studorlio).

**Dave Zhou** — AI builder. Empowered by AI, always solving real-world pain points.

- Live site: [https://zxzhou.github.io](https://zxzhou.github.io)
- LinkedIn: [linkedin.com/in/zxzhou](https://www.linkedin.com/in/zxzhou/)

## Blog (Jekyll)
- **Blog URL:** [https://zxzhou.github.io/blogs/](https://zxzhou.github.io/blogs/)
- Posts are Markdown files in `_posts/` (e.g. `_posts/2025-03-06-my-post.md`). Add a file, push to GitHub, and GitHub Pages will rebuild the site.

### Run the blog locally (Ruby 3.3 required)
**What goes wrong:** If you see `Could not find 'bundler' (4.0.3)` or `ruby 2.6`, the terminal is using **system Ruby**, not Homebrew Ruby 3.3. This repo’s Gemfile.lock was created with Ruby 3.3 and Bundler 4.

**Restart local test (in this order):**
1. Use Ruby 3.3 in this shell (do this first):
   ```bash
   export PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH"
   ```
2. Go to the site and install deps (if needed):
   ```bash
   cd zxzhou.github.io
   bundle install
   ```
3. Serve the site:
   ```bash
   bundle exec jekyll serve
   ```
4. Open **http://localhost:4000** (home) and **http://localhost:4000/blogs/** (blog).

To have Ruby 3.3 in every new terminal, add to `~/.zshrc`:  
`export PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH"` then run `source ~/.zshrc` or open a new terminal.

## Run locally (static only)
1. Open `index.html` in your browser (double-click or Right click → Open with → browser).
2. Or use a local server: `python3 -m http.server 8000` then visit http://localhost:8000

## FAQ
* [How do I fork this repository?](#how-do-i-fork-this-repository)
* [How do I rename the forked repository?](#how-do-i-rename-the-forked-repository)
* [How do I run the portfolio locally?](#how-do-i-run-the-portfolio-locally)

### How do I fork this repository?
1. Make sure you're logged into GitHub with your account
2. Click the Fork button on the upper right-hand side of this page

### How do I rename the forked repository?
1. Navigate to the main page of the repository. It should be `https://github.com/username/studorlio/`, where `username` is your GitHub username
2. Click Settings
3. Under the Repository name heading, type `username.github.io` then click Rename

### How do I run the portfolio locally?
1. Right click (Windows) or double click (Mac) `index.html` and select "Open with"
2. Pick your browser of choice

## Bugs and Issues
Have a bug or an issue with this template? [Open a new issue](https://github.com/helfi92/studorlio/issues)

## License
Code released under the [MIT](https://github.com/helfi92/studorlio/blob/master/LICENSE) license

