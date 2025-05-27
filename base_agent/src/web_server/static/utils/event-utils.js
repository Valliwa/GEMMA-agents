/**
 * Event-related utilities
 */

// Event type mapping for badges
export function getEventBadge(type) {
  const badges = {
    assistant_message: "assistant",
    tool_call: "tool",
    tool_result: "tool",
    agent_call: "agent",
    agent_result: "agent",
    overseer_notification: "overseer",
    system_prompt_update: "system",
    core_prompt_update: "system",
    default: "system",
  };
  return badges[type] || badges.default;
}

// Get node status indicator
export function getStatusIndicator(node) {
  if (!node.started_at) {
    return { class: "status-pending", label: "Pending" };
  }
  if (!node.completed_at) {
    return { class: "status-running", label: "Running" };
  }
  return node.success
    ? { class: "status-success", label: "Success" }
    : { class: "status-failed", label: "Failed" };
}

// Creates a chronological event stream from all events across all nodes
export function createChronologicalEventStream(nodes) {
  const allEvents = [];
  Object.entries(nodes).forEach(([nodeId, node]) => {
    if (node.events) {
      allEvents.push(
        ...node.events.map((event) => ({
          nodeId,
          nodeName: node.name,
          event,
          time: new Date(event.timestamp),
        })),
      );
    }
  });
  return allEvents.sort((a, b) => a.time - b.time);
}

// Sort events while maintaining agent call sequence
export function sortNodeEvents(events) {
  const sortedEvents = [];
  const tempEvents = [...events].sort(
    (a, b) => new Date(a.timestamp) - new Date(b.timestamp),
  );

  let i = 0;
  while (i < tempEvents.length) {
    const event = tempEvents[i];
    sortedEvents.push(event);
    i++;

    if (event.type === "agent_call") {
      const callTime = new Date(event.timestamp);
      const agentEvents = [];
      let j = i;
      let foundResult = false;
      while (j < tempEvents.length && !foundResult) {
        const nextEvent = tempEvents[j];
        if (
          nextEvent.type === "agent_result" &&
          new Date(nextEvent.timestamp) > callTime
        ) {
          agentEvents.push(nextEvent);
          tempEvents.splice(j, 1);
          foundResult = true;
          continue;
        }
        tempEvents.splice(j, 1);
        agentEvents.push(nextEvent);
      }
      sortedEvents.push(...agentEvents);
    }
  }

  return sortedEvents;
}

// UI interaction functions
export function toggleContent(index) {
  // Get event-stream component
  const eventStream = document.querySelector("event-stream");
  if (eventStream && eventStream.shadowRoot) {
    const truncated = eventStream.shadowRoot.querySelector(
      `#event-${index} .event-content`,
    );
    const full = eventStream.shadowRoot.querySelector(`#event-full-${index}`);
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
}

export function scrollToTop() {
  window.scrollTo({ top: 0, behavior: "smooth" });
}

export function scrollToStreamEvent(index) {
  // Get event-stream component
  const eventStream = document.querySelector("event-stream");
  if (eventStream && eventStream.shadowRoot) {
    const streamEvent = eventStream.shadowRoot.querySelector(`#event-${index}`);
    if (streamEvent) {
      streamEvent.scrollIntoView({ behavior: "smooth", block: "center" });
      streamEvent.classList.add("event-highlight");
      setTimeout(() => streamEvent.classList.remove("event-highlight"), 2000);

      // Expand the event details if needed
      const truncated = streamEvent.querySelector(`.event-content`);
      const full = streamEvent.querySelector(`#event-full-${index}`);
      if (truncated && full && truncated.classList.contains("hidden")) {
        toggleContent(index);
      }
    }
  }
}

export function toggleNode(nodeId) {
  const content = document.querySelector(`#${nodeId}-content`);
  if (content) {
    content.classList.toggle("hidden");
  }
}

// Expose required functions to window object for global access
window.scrollToStreamEvent = scrollToStreamEvent;
window.scrollToTop = scrollToTop;
window.toggleContent = toggleContent;
