<h2 id="Experiences">Experiences</h2>

<div class="experiences">
<ul class="bibliography">

{% for experience in site.data.experiences.main %}

<li>
<div class="pub-row">
  <div class="col-sm-3 abbr">
    {% if experience.logo %} 
    <img src="{{ experience.logo }}" class="teaser img-fluid z-depth-1">
    {% endif %}
  </div>
  <div class="col-sm-9">
      <div class="title">{{ experience.role }}{% if experience.company %} @ {{ experience.company }}{% endif %}</div>
      <div class="periodical">
        <em>
          {% if experience.start_date %}{{ experience.start_date }}{% endif %}
          {% if experience.end_date %} - {{ experience.end_date }}{% else %} - Present{% endif %}
        </em>
      </div>
      {% if experience.description %}
      <div class="author">{{ experience.description }}</div>
      {% endif %}
  </div>
</div>
</li>

{% endfor %}

</ul>
</div>
