[Hux Enhanced](kirisamer.github.io)
================================

> Based on Hux Blog, Enhanced with modern features.

![](https://github.com/KirisameR/KirisameR.github.io/blob/master/img/readme.jpg?raw=true)

[Enhancements]
--------------------------------------------------

1. Full $\LaTeX$ rendering support

2. Parameterized CSS, free to customize main/aux theme colors

3. Auto Light/Dark Mode

4. Parallax effect on header images

5. Open-box Busuanzi statistics (TODO)

6. One-key return top floating button (TODO)

7. Improved Theme style on `search`, `quote`, `tags`, `breadboard notification` and `bold text style`.

[User Manual 👉](_doc/Manual.md)
--------------------------------------------------

### Getting Started

1. You will need [Ruby](https://www.ruby-lang.org/en/) and [Bundler](https://bundler.io/) to use [Jekyll](https://jekyllrb.com/). Following [Using Jekyll with Bundler](https://jekyllrb.com/tutorials/using-jekyll-with-bundler/) to fullfill the enviromental requirement.

2. Installed dependencies in the `Gemfile`:

```sh
$ bundle install 
```

3. Serve the website (`localhost:4000` by default):

```sh
$ bundle exec jekyll serve  # alternatively, npm start
```

### Development (Build From Source)

To modify the theme, you will need [Grunt](https://gruntjs.com/). There are numbers of tasks you can find in the `Gruntfile.js`, includes minifing JavaScript, compiling `.less` to `.css`, adding banners to keep the Apache 2.0 license intact, watching for changes, etc. 

Yes, they were inherited and are extremely old-fashioned. There is no modularization and transpilation, etc.

Critical Jekyll-related code are located in `_include/` and `_layouts/`. Most of them are [Liquid](https://github.com/Shopify/liquid/wiki) templates.

This theme uses the default code syntax highlighter of jekyll, [Rouge](http://rouge.jneen.net/), which is compatible with Pygments theme so just pick any pygments theme css (e.g. from [here](http://jwarby.github.io/jekyll-pygments-themes/languages/javascript.html) and replace the content of `highlight.less`.


License
-------

Apache License 2.0.
Copyright (c) 2022-present KirisameR

Hux Enhanced is derived from [Clean Blog Jekyll Theme (MIT License)](https://github.com/BlackrockDigital/startbootstrap-clean-blog-jekyll/)
Copyright (c) 2013-2016 Blackrock Digital LLC.
