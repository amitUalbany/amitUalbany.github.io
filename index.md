---
layout: default
title: "Applied AI/ML Engineer"
description: "Portfolio for Amith Kumar Singh, PhD CS graduate focused on AI/ML, RAG, GraphRAG, graph learning, and production AI systems."
---

<section class="hero">
  <p class="eyebrow">PhD Computer Science - AI/ML - Open to roles</p>
  <h1>Applied AI/ML Engineer building reliable GenAI, GraphRAG, and graph learning systems.</h1>
  <p class="hero-copy">
    I am Amith Kumar Singh, a PhD CS graduate with 8+ years across software engineering,
    machine learning research, and production-oriented AI systems. My work connects rigorous
    statistical modeling with deployable ML products: retrieval systems, knowledge graphs,
    graph neural networks, probabilistic network diffusion, and evaluation pipelines.
  </p>
  <div class="hero-actions">
    <a class="button primary" href="{{ '/assets/resume/Amith_Kumar_Singh_Resume.pdf' | relative_url }}">Download Resume</a>
    <a class="button" href="https://www.linkedin.com/in/amith-kumar-singh-33568b68">LinkedIn</a>
    <a class="button" href="https://github.com/amitUalbany">GitHub</a>
    <a class="button" href="mailto:aks4amitkumarsingh@gmail.com">Email</a>
  </div>
</section>

<section class="quick-grid" aria-label="Career highlights">
  <div>
    <strong>Research depth</strong>
    <span>PhD work in statistical learning, graph theory, network diffusion, and sample complexity.</span>
  </div>
  <div>
    <strong>Production AI</strong>
    <span>FastAPI, Flask, Docker, structured logging, retrieval observability, and evaluation pipelines.</span>
  </div>
  <div>
    <strong>GenAI focus</strong>
    <span>RAG, GraphRAG, agentic workflows, tool calling, LangChain, LlamaIndex, RAGAS, and MCP.</span>
  </div>
</section>

<section id="projects" class="section">
  <div class="section-heading">
    <p class="eyebrow">Featured work</p>
    <h2>AI/ML projects recruiters can scan quickly</h2>
  </div>

  <div class="project-grid">
    <article class="project-card">
      <h3>GraphRAG Financial Intelligence Platform</h3>
      <p>
        End-to-end GraphRAG system over SEC 10-K filings combining vector search,
        knowledge graph traversal, reranking, and rank fusion for multi-hop financial QA.
      </p>
      <ul>
        <li>Improved Recall@10 by 12% versus vector-only RAG.</li>
        <li>Reduced retrieved context tokens by 15-19% with median retrieval latency around 280 ms.</li>
        <li>Built evaluation with RAGAS, Recall@K, answer relevance, latency, and observability metrics.</li>
      </ul>
      <p class="tags">Python - FastAPI - Qdrant - NetworkX - spaCy - Docker - OpenAI - RAGAS</p>
    </article>

    <article class="project-card">
      <h3>High-Precision RAG with Retrieval Observability</h3>
      <p>
        Hybrid RAG platform using BM25, dense retrieval, RRF, cross-encoders, deduplication,
        uncertainty signals, and governance thresholds for more trustworthy responses.
      </p>
      <ul>
        <li>Improved context precision by 20% over a vector-only baseline.</li>
        <li>Reduced redundant vectors by 35% with hash and semantic deduplication.</li>
        <li>Flagged low-confidence answers using entropy, stability, and logprob-based signals.</li>
      </ul>
      <p class="tags">BM25 - FAISS/Chroma-style retrieval - Cross-Encoders - LangChain - SciPy</p>
    </article>

    <article class="project-card">
      <h3>Statistical Learning for Network Diffusion</h3>
      <p>
        PhD dissertation project modeling opinion diffusion over attributed networks using
        probabilistic dynamics tied to graph topology, node attributes, and neighborhood structure.
      </p>
      <ul>
        <li>Established identifiability conditions and sample complexity guarantees.</li>
        <li>Designed efficient estimators for challenging parameter recovery settings.</li>
        <li>Evaluated diffusion behavior across random graphs, SBM, and scale-free networks.</li>
      </ul>
      <p class="tags">PyTorch - NumPy - NetworkX - MLE - Statistical Learning - Graph Theory</p>
    </article>

    <article class="project-card">
      <h3>AI Agent for Event Scheduling and Task Orchestration</h3>
      <p>
        Autonomous AI agent for multi-step execution using structured planning, tool calling,
        validation layers, and reliable downstream task orchestration.
      </p>
      <ul>
        <li>Implemented prompt chaining, confidence scoring, and Pydantic validation.</li>
        <li>Improved extraction accuracy by 20% through validation and error handling.</li>
        <li>Designed workflows for planning, tool use, and structured execution.</li>
      </ul>
      <p class="tags">Python - OpenAI API - Pydantic - MCP - Tool Calling</p>
    </article>
  </div>
</section>

<section class="section split">
  <div>
    <p class="eyebrow">Technical strengths</p>
    <h2>Built for AI roles that need both research judgment and shipping ability</h2>
  </div>
  <div class="skills-list">
    <p><strong>ML and GenAI:</strong> PyTorch, Scikit-learn, RAG, GraphRAG, LLM evaluation, prompt engineering, LangChain, LlamaIndex, LangSmith, RAGAS.</p>
    <p><strong>Retrieval and graph systems:</strong> semantic search, hybrid retrieval, cross-encoders, knowledge graphs, GCN/GAT, NetworkX, embeddings, rank fusion.</p>
    <p><strong>Engineering:</strong> Python, Java, SQL, FastAPI, Flask, REST APIs, Docker, PySpark, ETL pipelines, structured logging, reproducible workflows.</p>
    <p><strong>Research and evaluation:</strong> probabilistic modeling, hypothesis testing, A/B testing, benchmarking, error analysis, retrieval metrics.</p>
  </div>
</section>

<section class="section">
  <div class="section-heading">
    <p class="eyebrow">Selected publications</p>
    <h2>Research foundations</h2>
  </div>
  <div class="publication-list">
    <p><strong>Local Limit Theorems for Approximate Maximum Likelihood Estimation of Network Information Spreading Models.</strong> Magner, A., and Singh, A. K. IEEE ISIT, 2022.</p>
    <p><strong>Featurized Models of Network Opinion Spread: Identifiability and Sample Complexity.</strong> Singh, A. K., and Magner, A. University at Albany Research Showcase, 2024.</p>
    <p><strong>Sample Complexity Bounds for Estimation in Models of Network Averaging Dynamics.</strong> Magner, A., and Singh, A. K. IEEE ISIT Poster, 2025.</p>
  </div>
</section>

<section id="blog" class="section">
  <div class="section-heading row-heading">
    <div>
      <p class="eyebrow">Writing</p>
      <h2>Recent blog posts</h2>
    </div>
    <a class="text-link" href="{{ '/blog/' | relative_url }}">View all posts</a>
  </div>
  <div class="blog-list">
    {% for post in site.posts limit:3 %}
      <article>
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
        <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %-d, %Y" }}</time>
      </article>
    {% endfor %}
  </div>
</section>

<section id="contact" class="section contact-panel">
  <p class="eyebrow">Contact</p>
  <h2>Looking for Applied AI, ML Engineer, GenAI Engineer, or AI Research Engineer roles.</h2>
  <p>
    Based in Hillsborough, NJ. Available for conversations about AI/ML engineering,
    GenAI systems, RAG/GraphRAG, graph learning, and research-to-production roles.
  </p>
  <div class="hero-actions">
    <a class="button primary" href="mailto:aks4amitkumarsingh@gmail.com">aks4amitkumarsingh@gmail.com</a>
    <a class="button" href="https://www.linkedin.com/in/amith-kumar-singh-33568b68">LinkedIn</a>
    <a class="button" href="https://github.com/amitUalbany">GitHub</a>
  </div>
</section>
