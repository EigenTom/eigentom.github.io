<h2 id="Experiences" style="margin: 2px 0px 20px;">Experiences</h2>

<div class="experiences">
<ul class="bibliography" style="list-style-type: none; padding-left: 0;">

{% for experience in site.data.experiences.main %}

<li style="margin-bottom: 25px;">
<div class="pub-row" style="display: flex; align-items: center;">
  <div class="col-sm-3 abbr" style="position: relative; padding-right: 15px; padding-left: 40px; flex-shrink: 0; display: flex; align-items: center; justify-content: center;">
    {% if experience.logo %} 
    <img src="{{ experience.logo }}" class="teaser img-fluid z-depth-1" style="width: 100%; height: auto; max-width: 40px; max-height: 40px; object-fit: contain;">
    {% endif %}
  </div>
  <div class="col-sm-9" style="position: relative; padding-right: 15px; padding-left: 20px; flex: 1;">
      <div class="title">{{ experience.role }}{% if experience.company %} @ {{ experience.company }}{% endif %}</div>
      <div class="periodical">
        <em>
          {% if experience.start_date %}{{ experience.start_date }}{% endif %}
          {% if experience.end_date %} - {{ experience.end_date }}{% else %} - Present{% endif %}
        </em>
      </div>
      {% if experience.description %}
      <div class="author" style="margin-top: 8px;">{{ experience.description }}</div>
      {% endif %}
  </div>
</div>
</li>

{% endfor %}

</ul>
</div>

