<h2 id="news">News</h2>

<div class="news">
<div class="news-scroll">
<ul class="news-list">

{% for item in site.data.news.main %}

<li>
<div class="news-row">
  <div class="news-date">
    {{ item.date }}
  </div>
  <div class="news-content">
    {{ item.content }}
  </div>
</div>
</li>

{% endfor %}

</ul>
</div>
<div class="news-scroll-hint">Scroll for more updates</div>
</div>
