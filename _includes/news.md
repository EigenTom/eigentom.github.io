<h2 id="news" style="margin: 2px 0px 20px;">News</h2>

<div class="news">
<ul class="news-list" style="list-style-type: none; padding-left: 0;">

{% for item in site.data.news.main %}

<li style="margin-bottom: 15px;">
<div class="news-row" style="display: flex;">
  <div class="news-date" style="min-width: 100px; flex-shrink: 0; font-weight: bold; padding-right: -5px;">
    {{ item.date }}
  </div>
  <div class="news-content" style="flex: 1;">
    {{ item.content }}
  </div>
</div>
</li>

{% endfor %}

</ul>
</div>
