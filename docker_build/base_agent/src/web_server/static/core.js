/**
 * Core reactive system for the visualization
 */

// Base Component class
export class Component extends HTMLElement {
  constructor() {
    super();
    this.state = new Proxy(
      {},
      {
        set: (target, property, value) => {
          target[property] = value;
          this.render();
          return true;
        },
      },
    );
  }

  setState(newState) {
    Object.assign(this.state, newState);
  }

  render() {
    // Override in subclasses
  }

  connectedCallback() {
    this.render();
  }
}

// HTML escaping utility
export function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
