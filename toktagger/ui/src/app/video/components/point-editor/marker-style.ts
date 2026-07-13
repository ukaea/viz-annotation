/**
 * Point markers are drawn in screen pixels so they stay a constant size at any
 * zoom. The overlay and PointEditor.svelte must share these values, or points
 * will visibly resize when selected.
 */
export const POINT_MARKER = {
  ringRadiusPx: 9,
  selectedRingRadiusPx: 11,
  dotRadiusPx: 2.25,
  ringStrokePx: 2,
  selectedRingStrokePx: 2.5,
  dotStrokePx: 1.25,
  ringFill: "rgba(255, 255, 255, 0.12)",
  selectedRingFill: "rgba(56, 189, 248, 0.18)",
  ringStroke: "#ffffff",
  selectedRingStroke: "#38bdf8",
  dotFill: "#ffffff",
  dotStroke: "rgba(0, 0, 0, 0.9)",
} as const;
