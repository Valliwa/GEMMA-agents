/**
 * Formatting utilities
 */

export const formatters = {
  duration: (s) =>
    !s
      ? "0s"
      : s < 60
        ? `${s.toFixed(1)}s`
        : `${Math.floor(s / 60)}m ${(s % 60).toFixed(1)}s`,
  tokens: (t) =>
    !t
      ? "0"
      : t < 1000
        ? `${t}`
        : t < 1000000
          ? `${(t / 1000).toFixed(1)}K`
          : `${(t / 1000000).toFixed(1)}M`,
  cost: (c) =>
    !c
      ? "$0.00"
      : c < 0.01
        ? `$${c.toFixed(5)}`
        : c < 0.1
          ? `$${c.toFixed(4)}`
          : c < 1
            ? `$${c.toFixed(3)}`
            : `$${c.toFixed(2)}`,
  cachePercent: (cached, total) =>
    !total ? "0%" : `${((cached / total) * 100).toFixed(1)}% cached`,
};

// Get total tokens from usage object
export function getTotalTokens(usage) {
  if (!usage) return 0;
  return (
    (usage.uncached_prompt_tokens || 0) +
    (usage.cache_write_prompt_tokens || 0) +
    (usage.cached_prompt_tokens || 0) +
    (usage.completion_tokens || 0)
  );
}
