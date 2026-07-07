---
layout: default
title: "Blog"
description: "Technical writing by Amith Kumar Singh on AI, ML, model deployment, and applied research."
permalink: /blog/
---

<section class="page-title">
  <p class="eyebrow">Blog</p>
  <h1>Technical writing</h1>
  <p>Notes on AI/ML systems, model deployment, retrieval, quantization, and applied research.</p>
</section>

<section class="blog-list full">
  {% for post in site.posts %}
    <article>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %-d, %Y" }}</time>
      {% if post.excerpt %}
        <p>{{ post.excerpt | strip_html | truncate: 180 }}</p>
      {% endif %}
    </article>
  {% endfor %}
</section>
