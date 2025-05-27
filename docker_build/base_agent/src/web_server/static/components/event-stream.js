/**
 * EventStream component
 */

import { Component, escapeHtml } from "../core.js";
import { formatters, getTotalTokens } from "../utils/formatters.js";
import { createChronologicalEventStream } from "../utils/event-utils.js";
import { store } from "../store.js";

export class EventStream extends Component {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    // Track expanded state
    this.expandedEvents = new Set();

    // Add styles
    const style = document.createElement("style");
    style.textContent = `
      :host {
        display: block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        line-height: 1.3;
      }
      .event {
        margin-bottom: 1rem;
        border-radius: 0.25rem;
        overflow: hidden;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
      }
      .event-header {
        padding: 0.75rem;
        background-color: #f9fafb;
        position: relative;
        display: flex;
        flex-direction: column;
      }
      /* Vertical lines for different event types */
      .event-header::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
      }
      .event.assistant-message .event-header::before {
        background-color: #3b82f6;  /* blue */
      }
      .event.tool-call .event-header::before,
      .event.tool-result .event-header::before {
        background-color: #8b5cf6;  /* purple */
      }
      .event.agent-call .event-header::before,
      .event.agent-result .event-header::before {
        background-color: #f97316;  /* orange */
      }
      .event.overseer-notification .event-header::before {
        background-color: #ef4444;  /* red */
      }
      .event.system-prompt-update .event-header::before,
      .event.core-prompt-update .event-header::before {
        background-color: #374151;  /* gray */
      }
      .event-content,
      .event-full-content {
        background-color: #f8fafc;
        transition: background-color 0.2s;
        padding: 0.75rem;
      }
      .event-content:hover,
      .event-full-content:hover {
        background-color: #f1f5f9;
      }
      .up-arrow {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #e5e7eb;
        background-color: transparent;
        color: #6b7280;
        font-size: 1rem;
        line-height: 1;
        cursor: pointer;
        transition: all 0.2s;
      }
      .up-arrow:hover {
        background-color: #f3f4f6;
        color: #374151;
        border-color: #d1d5db;
      }
      .flex { display: flex; }
      .flex-col { flex-direction: column; }
      .items-center { align-items: center; }
      .justify-between { justify-content: space-between; }
      .font-medium { font-weight: 500; }
      .text-gray-700 { color: #374151; }
      .text-gray-500 { color: #6b7280; }
      .text-gray-400 { color: #9ca3af; }
      .text-xs { font-size: 0.75rem; }
      .mt-1 { margin-top: 0.25rem; }
      .ml-2 { margin-left: 0.5rem; }
      .font-mono { font-family: 'JetBrains Mono', monospace; }
      .whitespace-pre-wrap { white-space: pre-wrap; }
      .cursor-pointer { cursor: pointer; }
      .hidden { display: none !important; }
      .italic { font-style: italic; }
    `;
    this.shadowRoot.appendChild(style);

    // Create container
    this.container = document.createElement("div");
    this.shadowRoot.appendChild(this.container);

    // Bind event handlers
    this.handleContentToggle = this.handleContentToggle.bind(this);

    // Listen for state changes
    document.addEventListener("state-change", (e) => {
      if (e.detail.property === "callgraphData") {
        this.setState({ data: e.detail.value });
      }
    });
  }

  handleContentToggle(index) {
    if (this.expandedEvents.has(index)) {
      this.expandedEvents.delete(index);
    } else {
      this.expandedEvents.add(index);
    }
    const truncated = this.shadowRoot.querySelector(
      `#event-${index} .event-content`,
    );
    const full = this.shadowRoot.querySelector(`#event-full-${index}`);
    if (truncated && full) {
      if (truncated.classList.contains("hidden")) {
        truncated.classList.remove("hidden");
        full.classList.add("hidden");
      } else {
        truncated.classList.add("hidden");
        full.classList.remove("hidden");
      }
    }
  }

  createEventStreamView(eventStream) {
    if (!eventStream.length) {
      return '<div class="text-gray-500 text-center py-8">No events recorded</div>';
    }

    return eventStream
      .map((item, index) => {
        const { event, nodeName, nodeId } = item;

        let metrics = [];
        if (event.metadata && event.metadata.completion) {
          const completion = event.metadata.completion;
          if (completion.usage) {
            const totalTokens = getTotalTokens(completion.usage);
            const cachedTokens = completion.usage.cached_prompt_tokens || 0;
            metrics.push(`${totalTokens} tokens`);
            if (cachedTokens > 0) {
              metrics.push(
                `${formatters.cachePercent(cachedTokens, totalTokens)}`,
              );
            }
          }
          metrics.push(formatters.cost(completion.cost || 0));
        }

        const lines = event.content.split("\n");
        const truncatedLines = lines.slice(0, 20).join("\n");

        // Further truncate by length
        const maxCharacters = 2000;
        const truncated =
          truncatedLines.length > maxCharacters
            ? truncatedLines.slice(0, maxCharacters) + "..."
            : truncatedLines;

        const hiddenLines = Math.max(0, lines.length - 20);
        const foldedIndicator =
          hiddenLines > 0
            ? `\n\n<span class="text-gray-500 italic">${hiddenLines} more line${hiddenLines > 1 ? "s" : ""} hidden</span>`
            : "";

        let name = "";
        if (event.type === "tool_call" || event.type === "tool_result") {
          name = event.metadata.name ? event.metadata.name : "";
        }

        const isExpanded = this.expandedEvents.has(index);

        // Eliminate template literal indentation by using array join
        return [
          `<div id="event-${index}" class="event ${event.type.replace("_", "-")}">`,
          `  <div class="event-header">`,
          `    <div class="flex items-center justify-between">`,
          `      <div class="flex items-center">`,
          `        <span class="font-medium text-gray-700">${escapeHtml(event.type)} ${name}</span>`,
          `        <span class="ml-2 text-gray-500">[${escapeHtml(nodeName)}:${escapeHtml(nodeId)}]</span>`,
          `      </div>`,
          `      <div class="flex items-center">`,
          `        <span class="text-xs text-gray-400">${new Date(event.timestamp).toISOString().split("T")[1].split(".")[0]}</span>`,
          `        <button class="up-arrow ml-2" onclick="window.scrollToTop()" title="Back to top">↑</button>`,
          `      </div>`,
          `    </div>`,
          `    ${metrics.length ? `<div class="text-xs text-gray-500 mt-1">${metrics.join(" | ")}</div>` : ""}`,
          `  </div>`,
          `  <div class="event-content font-mono whitespace-pre-wrap cursor-pointer ${isExpanded ? "hidden" : ""}" onclick="this.getRootNode().host.handleContentToggle(${index})">${escapeHtml(truncated)}${foldedIndicator}</div>`,
          `  <div id="event-full-${index}" class="event-full-content font-mono whitespace-pre-wrap cursor-pointer ${!isExpanded ? "hidden" : ""}" onclick="this.getRootNode().host.handleContentToggle(${index})">${escapeHtml(event.content)}</div>`,
          `</div>`,
        ].join("\n");
      })
      .join("");
  }

  render() {
    const data = this.state.data;
    if (!data || !data.root_id || !data.nodes) {
      this.container.innerHTML =
        '<div class="text-gray-500 text-center py-8">No events recorded</div>';
      return;
    }

    const rootNode = data.nodes[data.root_id];
    if (rootNode) {
      const allEvents = createChronologicalEventStream(data.nodes);
      this.container.innerHTML = this.createEventStreamView(allEvents);
    }
  }
}

customElements.define("event-stream", EventStream);
