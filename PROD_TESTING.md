**Production Testing & Security Considerations**

- **Testing / Experimentation Approaches**
  - Shadow deployment: Duplicate live traffic to a candidate model running in parallel (no user-visible effect). Compare predictions and metrics offline to detect regressions before full rollout.
  - Canary release: Gradually route a small percentage (e.g., 1-5%) of real traffic to the new model version, monitor errors and quality metrics, then increase traffic if safe.
  - A/B testing: Randomly split traffic between baseline and candidate models to measure business metrics (e.g., revenue, click-through) and statistical significance.

- **Monitoring & Safety Gates**
  - Define automated SLOs/thresholds (e.g., daily MAE drift, latency, error rate) that trigger rollbacks or human review.
  - Log prediction inputs, outputs, and model version for auditing and reproducibility. Store a sample of requests for offline analysis and drift detection.

- **Security Consideration (one concrete example)**n+  - Input validation & rate limiting: Always validate request schema (done via Pydantic) and apply request throttling and authentication. Reject or rate-limit suspicious traffic patterns to reduce exposure to adversarial or resource-exhaustion attacks.

This document is intentionally short — expand with runbooks and monitoring playbooks when moving to production.
