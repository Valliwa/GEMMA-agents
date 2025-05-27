/**
 * Main visualization entry point
 */

import { startUpdates } from "./store.js";
import "./components/execution-tree.js";
import "./components/event-stream.js";
import "./components/metrics-display.js";
import {
  toggleContent,
  toggleNode,
  scrollToTop,
  scrollToStreamEvent,
} from "./utils/event-utils.js";

// Make UI functions globally available
window.toggleContent = toggleContent;
window.toggleNode = toggleNode;
window.scrollToTop = scrollToTop;
window.scrollToStreamEvent = scrollToStreamEvent;

// Start updates
startUpdates();
