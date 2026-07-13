{% assign mapmyvisitors_light_ocean = "f7f8fa" %}
{% assign mapmyvisitors_light_land = "0a6ca8" %}
{% assign mapmyvisitors_light_past = "f1b7b7" %}
{% assign mapmyvisitors_light_recent = "b8e8b8" %}
{% assign mapmyvisitors_light_text = "16191f" %}
{% assign mapmyvisitors_dark_ocean = "1f2228" %}
{% assign mapmyvisitors_dark_land = "0a6ca8" %}
{% assign mapmyvisitors_dark_past = "7f2f2f" %}
{% assign mapmyvisitors_dark_recent = "2f7f3a" %}
{% assign mapmyvisitors_dark_text = "ffffff" %}
<script type="text/javascript">
	(function() {
		var mediaQuery = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

		function buildParams(isDark) {
			return isDark
				? "cl={{ mapmyvisitors_dark_land }}&co={{ mapmyvisitors_dark_ocean }}&cmo={{ mapmyvisitors_dark_past }}&cmn={{ mapmyvisitors_dark_recent }}&ct={{ mapmyvisitors_dark_text }}"
				: "cl={{ mapmyvisitors_light_land }}&co={{ mapmyvisitors_light_ocean }}&cmo={{ mapmyvisitors_light_past }}&cmn={{ mapmyvisitors_light_recent }}&ct={{ mapmyvisitors_light_text }}";
		}

		var script = document.createElement("script");
		var isDark = mediaQuery && mediaQuery.matches;

		script.type = "text/javascript";
		script.id = "mapmyvisitors";
		script.src = "https://mapmyvisitors.com/map.js?" + buildParams(isDark) + "&w=a&t=m&d=aHCMdlI2tDN-xAebM-gf37yJNxg9UMxmIFPrWPcuqsM";
		document.currentScript.parentNode.insertBefore(script, document.currentScript);

		if (mediaQuery) {
			var reloadOnThemeChange = function() {
				window.location.reload();
			};

			if (typeof mediaQuery.addEventListener === "function") {
				mediaQuery.addEventListener("change", reloadOnThemeChange);
			} else if (typeof mediaQuery.addListener === "function") {
				mediaQuery.addListener(reloadOnThemeChange);
			}
		}
	})();
</script>