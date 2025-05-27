/**
 * MetricsDisplay component
 */

import { Component } from "../core.js";
import { formatters } from "../utils/formatters.js";
import { store } from "../store.js";

export class MetricsDisplay extends Component {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });

    // Add styles
    const style = document.createElement("style");
    style.textContent = `
      :host {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        color: white;
      }
      .metric {
        display: flex;
        align-items: center;
        margin-right: 1.5rem;
      }
      .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #d1d5db;
      }
      .value {
        margin-left: 0.5rem;
        font-size: 0.875rem;
        font-weight: 500;
      }
      .cached {
        font-size: 0.75rem;
        color: #d1d5db;
      }
    `;
    this.shadowRoot.appendChild(style);

    // Create container
    this.container = document.createElement("div");
    this.container.style.display = "flex";
    this.container.style.flexWrap = "wrap";
    this.container.style.alignItems = "center";
    this.shadowRoot.appendChild(this.container);

    // Listen for state changes
    document.addEventListener("state-change", (e) => {
      if (e.detail.property === "callgraphData") {
        this.setState({ data: e.detail.value });
      }
    });
  }

  render() {
    const data = this.state.data || {};

    this.container.innerHTML = `
      <div class="metric">
        <span class="label">Duration</span>
        <span class="value">${formatters.duration(data.total_duration)}</span>
      </div>
      <div class="metric">
        <span class="label">Total Tokens</span>
        <span class="value">${formatters.tokens(data.total_tokens)}</span>
        <span class="cached">${data.total_tokens ? formatters.cachePercent(data.num_cached_tokens, data.total_tokens) : "-"}</span>
      </div>
      <div class="metric">
        <span class="label">Cost</span>
        <span class="value">${formatters.cost(data.total_cost)}</span>
      </div>
    `;
  }
}

customElements.define("metrics-display", MetricsDisplay);
