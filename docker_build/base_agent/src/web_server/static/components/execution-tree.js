/**
 * ExecutionTree component
 */

import { Component, escapeHtml } from "../core.js";
import { formatters, getTotalTokens } from "../utils/formatters.js";
import {
  getStatusIndicator,
  createChronologicalEventStream,
} from "../utils/event-utils.js";
import { store } from "../store.js";

export class ExecutionTree extends Component {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });

    // Track collapsed nodes using actual node IDs from data
    this.collapsedNodes = new Set();

    // Add styles
    const style = document.createElement("style");
    style.textContent = `
      :host {
        display: block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        line-height: 1.3;
      }
      .node {
        margin-bottom: 0.25rem;
        position: relative;
      }
      .node-header {
        padding: 0.5rem;
        background-color: #f2f3f5;
        border-radius: 0.25rem;
        cursor: pointer;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: flex-start;
        flex-direction: column;
      }
      .node-info {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.25rem;
      }
      .node-stats {
        margin-top: 0.25rem;
        padding-left: calc(10px + 0.5rem); /* Status circle width + margin */
      }
      .node-content {
        margin-left: 0.90rem;
        position: relative;
        padding-left: 12px; /* Space for the line */
        transition: height 0.2s ease-out;
        overflow: hidden;
      }
      .node-content::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 2px;
        border-radius: 5px;
        background-color: #e5e7eb; /* Gray line */
      }
      .node-header:hover {
        background-color: #e5e7eb;
      }
      .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
        position: relative;
        z-index: 3;
      }
      .status-pending { background-color: #fbbf24; }
      .status-running {
        background-color: #60a5fa;
        animation: pulse 2s infinite;
      }
      .status-success { background-color: #34d399; }
      .status-failed { background-color: #f87171; }
      @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
      }
      .event-line {
        padding-top: 0.125rem;
        padding-bottom: 0.125rem;
        border-radius: 0.25rem;
        // transition: background-color 0.2s ease;
      }
      .event-line:hover {
        background-color: #e1eaf5; /* Light blue background */
        cursor: pointer;
      }
      .text-blue-500 { color: #3b82f6; }
      .text-purple-500 { color: #8b5cf6; }
      .text-orange-500 { color: #f97316; }
      .text-red-500 { color: #ef4444; }
      .text-gray-500 { color: #6b7280; }
      .text-gray-700 { color: #374151; }
      .ml-2 { margin-left: 0.5rem; }
      .whitespace-nowrap { white-space: nowrap; }
      .truncate {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .flex { display: flex; }
      .items-center { align-items: center; }
      .text-xs { font-size: 0.75rem; }
      .text-sm { font-size: 0.875rem; }
      .font-medium { font-weight: 500; }
      .ml-1 { margin-left: 0.25rem; }
      .cursor-pointer { cursor: pointer; }
      .space-x-2 > * + * { margin-left: 0.5rem; }
      .hidden { display: none; }
    `;
    this.shadowRoot.appendChild(style);

    // Create container
    this.container = document.createElement("div");
    this.shadowRoot.appendChild(this.container);

    // Bind event handlers
    this.toggleNode = this.toggleNode.bind(this);

    // Listen for state changes
    document.addEventListener("state-change", (e) => {
      if (e.detail.property === "callgraphData") {
        this.setState({ data: e.detail.value });
      }
    });
  }

  toggleNode(nodeId) {
    if (this.collapsedNodes.has(nodeId)) {
      this.collapsedNodes.delete(nodeId);
    } else {
      this.collapsedNodes.add(nodeId);
    }
    const content = this.shadowRoot.querySelector(`#node-content-${nodeId}`);
    if (content) {
      content.classList.toggle("hidden");
    }
  }

  formatTreeEvent(event, startTime, globalEventIndex, functionLevel) {
    const eventTime = new Date(event.timestamp);
    const relativeTime = ((eventTime - startTime) / 1000).toFixed(1);
    const indentClass = functionLevel > 0 ? `ml-${functionLevel * 4}` : "";

    let eventContent = "";
    switch (event.type) {
      case "assistant_message":
        const metadata = event.metadata || {};
        const completion = metadata.completion;
        let metrics = [];
        metrics.push(`t+${relativeTime}s`);
        if (completion) {
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
          if (completion.timing && completion.timing.time_to_first_token) {
            metrics.push(
              `ttft: ${completion.timing.time_to_first_token.toFixed(2)}s`,
            );
          }
        }
        eventContent = `
          <span class="text-blue-500">assistant</span>
          ${metrics.length ? `<span class="text-gray-500 whitespace-nowrap">${metrics.join(" | ")}</span>` : ""}
          ${event.content ? `<span class="text-gray-700 truncate">"${escapeHtml(event.content.substring(0, 80))}${event.content.length > 80 ? "..." : ""}"</span>` : ""}
        `;
        break;
      case "tool_call":
        const toolName = event.metadata?.name || "unknown tool";
        const toolArgs = JSON.stringify(event.metadata?.args || {});
        eventContent = `
          <span class="text-purple-500">tool call</span>
          <span class="ml-2 text-gray-700">${escapeHtml(toolName)}</span>
          <span class="ml-2 text-gray-500">| t+${relativeTime}s</span>
          <span class="ml-2 text-gray-700 truncate">${escapeHtml(toolArgs.substring(0, 50))}${toolArgs.length > 50 ? "..." : ""}</span>
        `;
        break;
      case "tool_result":
        const resultName =
          event.metadata?.tool_result?.tool_name || "unknown tool";
        const success =
          event.metadata?.tool_result?.success !== undefined
            ? event.metadata.tool_result.success
            : true;
        const duration = event.metadata?.tool_result?.duration
          ? `${event.metadata.tool_result.duration.toFixed(1)}s`
          : "0.0s";
        eventContent = `
          <span class="text-purple-500">tool result</span>
          <span class="ml-2 text-gray-700">${escapeHtml(resultName)}</span>
          <span class="ml-2 text-gray-500">| ${duration} → ${success ? "Success" : "Failed"}</span>
        `;
        break;
      case "agent_call":
        const agentName = event.metadata?.name || "unknown agent";
        const agentArgs = JSON.stringify(event.metadata?.args || {});
        eventContent = `
          <span class="text-orange-500">agent call</span>
          <span class="ml-2 text-gray-700">${escapeHtml(agentName)}</span>
          <span class="ml-2 text-gray-500">| t+${relativeTime}s | </span>
          <span class="text-gray-700 truncate">${escapeHtml(agentArgs.substring(0, 50))}${agentArgs.length > 50 ? "..." : ""}</span>
        `;
        break;
      case "agent_result":
        const resultAgentName =
          event.metadata?.agent_result?.agent_name || "unknown agent";
        const agentSuccess =
          event.metadata?.agent_result?.success !== undefined
            ? event.metadata.agent_result.success
            : true;
        const agentDuration = event.metadata?.agent_result?.metrics
          ?.duration_seconds
          ? `${event.metadata.agent_result.metrics.duration_seconds.toFixed(1)}s`
          : "0.0s";
        eventContent = `
          <span class="text-orange-500">agent result</span>
          <span class="ml-2 text-gray-700">${escapeHtml(resultAgentName)}</span>
          <span class="ml-2 text-gray-500">| ${agentDuration} → ${agentSuccess ? "Success" : "Failed"}</span>
        `;
        break;
      case "system_prompt_update":
      case "core_prompt_update":
        const lines = event.content.split("\n").length;
        const type = event.type === "system_prompt_update" ? "system" : "core";
        eventContent = `
          <span class="text-gray-500">${type}</span>
          <span class="ml-2 text-gray-700">prompt update</span>
          <span class="ml-2 text-gray-500">| t+${relativeTime}s | ${lines} lines</span>
        `;
        break;
      case "overseer_notification":
        eventContent = `
          <span class="text-red-500">overseer</span>
          <span class="ml-2 text-gray-500">| t+${relativeTime}s</span>
          <span class="ml-2 text-gray-700 truncate">${escapeHtml(event.content.substring(0, 80))}${event.content.length > 80 ? "..." : ""}</span>
        `;
        break;
      default:
        eventContent = `
          <span class="text-gray-500">${event.type}</span>
          <span class="ml-2 text-gray-500">| t+${relativeTime}s</span>
          <span class="ml-2 text-gray-700 truncate">${escapeHtml(event.content.substring(0, 80))}${event.content.length > 80 ? "..." : ""}</span>
        `;
    }

    return `
      <div class="event-line ${indentClass}" onclick="window.scrollToStreamEvent(${globalEventIndex})">
        <div class="flex items-center text-xs space-x-2">
          ${eventContent}
        </div>
      </div>
    `;
  }

  createExecutionTree(rootNode, allNodes) {
    const startTime = new Date(rootNode.started_at || Date.now());
    let nodeIdCounter = 0;

    // Get all events in chronological order with their node info
    const allEvents = createChronologicalEventStream(allNodes);

    // Create a map of events to their global index
    const eventIndexMap = new Map(
      allEvents.map((item, index) => [item.event, index]),
    );

    const formatNode = (node, level = 0) => {
      const indent = level * 8;
      let html = "";
      const nodeId = node.id; // Use actual node ID from data

      // Node metrics and stats
      const metrics = [];
      if (node.duration_seconds)
        metrics.push(formatters.duration(node.duration_seconds));
      if (node.token_count) {
        metrics.push(`${formatters.tokens(node.token_count)} tokens`);
        if (node.num_cached_tokens) {
          metrics.push(
            formatters.cachePercent(node.num_cached_tokens, node.token_count),
          );
        }
      }
      if (node.cost) metrics.push(formatters.cost(node.cost));

      let stats = "";
      if (node.events) {
        const eventCounts = node.events.reduce((acc, e) => {
          acc[e.type] = (acc[e.type] || 0) + 1;
          return acc;
        }, {});

        let statsLine = "Events: ";
        const statsList = [];
        if (eventCounts.system_prompt_update) {
          const lines = node.events
            .filter((e) => e.type === "system_prompt_update")
            .reduce((acc, e) => acc + e.content.split("\n").length, 0);
          statsList.push(
            `${eventCounts.system_prompt_update} system updates (${lines} lines)`,
          );
        }
        if (eventCounts.core_prompt_update) {
          const lines = node.events
            .filter((e) => e.type === "core_prompt_update")
            .reduce((acc, e) => acc + e.content.split("\n").length, 0);
          statsList.push(
            `${eventCounts.core_prompt_update} core updates (${lines} lines)`,
          );
        }
        if (eventCounts.tool_call)
          statsList.push(`${eventCounts.tool_call} tool calls`);
        if (eventCounts.assistant_message)
          statsList.push(`${eventCounts.assistant_message} messages`);
        if (statsList.length) {
          stats = `${statsLine}${statsList.join(", ")}`;
        }
      }

      // Check if this node should be collapsed using node.id
      const isCollapsed = this.collapsedNodes.has(nodeId);

      // Node header with all info
      const status = getStatusIndicator(node);
      html += `
        <div class="node" style="margin-left: ${indent}px" id="node-${nodeId}">
          <div class="node-header" onclick="this.getRootNode().host.toggleNode('${nodeId}')">
            <div class="node-info">
              <span class="status-indicator ${status.class}" title="${status.label}"></span>
              <span class="font-medium text-sm">${escapeHtml(node.name)}</span>
              <span class="text-gray-500">[${node.id}]</span>
              ${metrics.length ? `<span class="text-xs text-gray-500">(${metrics.join(" | ")})</span>` : ""}
            </div>
            ${stats ? `<div class="node-stats text-xs text-gray-500">${stats}</div>` : ""}
          </div>
          <div class="node-content ${isCollapsed ? "hidden" : ""}" id="node-content-${nodeId}">
      `;

      if (node.events) {
        // Create timeline of events and child functions
        const timeline = [];
        let functionLevel = 0;

        // Add events
        node.events.forEach((event) => {
          const globalIndex = eventIndexMap.get(
            allEvents.find(
              (item) => item.nodeId === node.id && item.event === event,
            )?.event,
          );
          timeline.push({
            time: new Date(event.timestamp),
            type: "event",
            event,
            globalIndex,
            level: functionLevel,
          });
        });

        // Add child functions with start and end
        if (node.children) {
          node.children.forEach((childId) => {
            const childNode = allNodes[childId];
            if (childNode && childNode.started_at) {
              timeline.push({
                time: new Date(childNode.started_at),
                type: "function_start",
                node: childNode,
              });
              if (childNode.completed_at) {
                timeline.push({
                  time: new Date(childNode.completed_at),
                  type: "function_end",
                  node: childNode,
                });
              }
            }
          });
        }

        // Sort timeline
        timeline.sort((a, b) => a.time - b.time);

        // Process timeline
        for (const item of timeline) {
          if (item.type === "event") {
            html += this.formatTreeEvent(
              item.event,
              startTime,
              item.globalIndex,
              functionLevel,
            );
          } else if (item.type === "function_start") {
            functionLevel++;
            html += formatNode(item.node, level + 1);
          } else if (item.type === "function_end") {
            functionLevel--;
          }
        }
      }

      html += `</div></div>`;
      return html;
    };

    return formatNode(rootNode);
  }

  render() {
    const data = this.state.data;
    if (!data || !data.root_id || !data.nodes) {
      this.container.innerHTML =
        '<div class="text-gray-500 text-center py-8">No active execution</div>';
      return;
    }

    const rootNode = data.nodes[data.root_id];
    if (rootNode) {
      this.container.innerHTML = this.createExecutionTree(rootNode, data.nodes);
    }
  }
}

customElements.define("execution-tree", ExecutionTree);
