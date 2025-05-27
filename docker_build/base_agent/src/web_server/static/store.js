/**
 * Centralized state management with WebSocket support
 */

export const store = new Proxy(
  {
    callgraphData: null,
  },
  {
    set(target, property, value) {
      target[property] = value;
      document.dispatchEvent(
        new CustomEvent("state-change", {
          detail: { property, value },
        })
      );
      return true;
    },
  }
);

let socket;

export async function updateVisualization() {
  try {
    const response = await fetch("/api/callgraph");
    const data = await response.json();

    // Skip if data hasn't changed
    if (JSON.stringify(data) !== JSON.stringify(store.callgraphData)) {
      store.callgraphData = data;
    }
  } catch (error) {
    console.error("Error updating visualization:", error);
  }
}

function connectWebSocket() {
  socket = new WebSocket(`ws://${window.location.host}/ws`);

  socket.onopen = () => {
    console.log("WebSocket connected");
  };

  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'event') {
      // Get latest data to incorporate the new event
      updateVisualization();
    }
  };

  socket.onclose = () => {
    console.log("WebSocket disconnected. Reconnecting...");
    setTimeout(connectWebSocket, 1000);
  };

  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
  };
}

// Start WebSocket connection and initial data load
export function startUpdates() {
  updateVisualization(); // Initial load
  connectWebSocket();    // Real-time updates
}