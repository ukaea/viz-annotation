<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import type { Ellipse, Transform } from "@annotorious/annotorious";
  import { POINT_MARKER } from "./marker-style";

  type PointEditorEvents = {
    change: Ellipse;
    grab: PointerEvent;
    release: PointerEvent;
  };

  export let shape: Ellipse;
  // Annotorious pushes this prop to registered editors. Point marker styling is
  // intentionally fixed here so it matches PointCenterDotOverlay exactly.
  export let computedStyle: string | undefined;
  export let transform: Transform;
  export let viewportScale = 1;
  export let svgEl: SVGSVGElement | undefined;

  const dispatch = createEventDispatcher<PointEditorEvents>();

  let grabbedPointerId: number | null = null;
  let origin: [number, number] | null = null;
  let initialShape: Ellipse | null = null;
  let captureTarget: Element | null = null;

  $: geom = shape.geometry;
  $: scale = Math.max(Number(viewportScale) || 1, 0.0001);
  $: ringRadius = POINT_MARKER.selectedRingRadiusPx / scale;
  $: markerRadius = POINT_MARKER.dotRadiusPx / scale;
  $: hitRx = ringRadius;
  $: hitRy = ringRadius;

  const eventToImagePoint = (evt: PointerEvent): [number, number] => {
    if (svgEl) {
      const { left, top } = svgEl.getBoundingClientRect();
      return transform.elementToImage(evt.clientX - left, evt.clientY - top);
    }

    return transform.elementToImage(evt.offsetX, evt.offsetY);
  };

  const translateShape = (
    ellipse: Ellipse,
    dx: number,
    dy: number,
  ): Ellipse => {
    const { geometry } = ellipse;

    return {
      ...ellipse,
      geometry: {
        ...geometry,
        cx: geometry.cx + dx,
        cy: geometry.cy + dy,
        bounds: {
          minX: geometry.bounds.minX + dx,
          minY: geometry.bounds.minY + dy,
          maxX: geometry.bounds.maxX + dx,
          maxY: geometry.bounds.maxY + dy,
        },
      },
    };
  };

  const grabShape = (evt: PointerEvent) => {
    if (evt.button !== 0) return;

    grabbedPointerId = evt.pointerId;
    origin = eventToImagePoint(evt);
    initialShape = shape;
    captureTarget =
      evt.currentTarget instanceof Element
        ? evt.currentTarget
        : evt.target instanceof Element
          ? evt.target
          : null;

    captureTarget?.setPointerCapture(evt.pointerId);
    dispatch("grab", evt);
  };

  const moveShape = (evt: PointerEvent) => {
    if (
      grabbedPointerId !== evt.pointerId ||
      origin === null ||
      initialShape === null
    ) {
      return;
    }

    const [x, y] = eventToImagePoint(evt);
    shape = translateShape(initialShape, x - origin[0], y - origin[1]);
    dispatch("change", shape);
  };

  const releaseShape = (evt: PointerEvent) => {
    if (grabbedPointerId !== evt.pointerId) return;

    if (captureTarget?.hasPointerCapture(evt.pointerId)) {
      captureTarget.releasePointerCapture(evt.pointerId);
    }

    grabbedPointerId = null;
    origin = null;
    initialShape = null;
    captureTarget = null;

    dispatch("release", evt);
  };
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<g
  class="a9s-annotation selected point-editor"
  on:pointermove={moveShape}
  on:pointerup={releaseShape}
  on:pointercancel={releaseShape}
>
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <ellipse
    class="point-hit-target a9s-shape-handle"
    on:pointerdown={grabShape}
    cx={geom.cx}
    cy={geom.cy}
    rx={hitRx}
    ry={hitRy}
  />

  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <ellipse
    class="point-marker-ring a9s-shape-handle"
    on:pointerdown={grabShape}
    cx={geom.cx}
    cy={geom.cy}
    rx={ringRadius}
    ry={ringRadius}
    style="--ring-fill: {POINT_MARKER.selectedRingFill};
      --ring-stroke: {POINT_MARKER.selectedRingStroke};
      --ring-stroke-width: {POINT_MARKER.selectedRingStrokePx}px;"
  />

  <circle
    class="point-center-dot"
    cx={geom.cx}
    cy={geom.cy}
    r={markerRadius}
    style="--dot-fill: {POINT_MARKER.dotFill};
      --dot-stroke: {POINT_MARKER.dotStroke};
      --dot-stroke-width: {POINT_MARKER.dotStrokePx}px;"
  />
</g>

<style>
  .point-hit-target {
    fill: transparent;
    pointer-events: all;
    stroke: transparent;
    stroke-width: 0;
  }

  .point-marker-ring {
    /* Annotorious makes editor-layer ellipses transparent. Scoped CSS has
       enough specificity to override it; SVG presentation attributes do not. */
    fill: var(--ring-fill);
    pointer-events: all;
    stroke: var(--ring-stroke);
    stroke-width: var(--ring-stroke-width);
    vector-effect: non-scaling-stroke;
  }

  .point-center-dot {
    fill: var(--dot-fill);
    pointer-events: none;
    stroke: var(--dot-stroke);
    stroke-width: var(--dot-stroke-width);
    vector-effect: non-scaling-stroke;
  }
</style>
