<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import type { Ellipse, Transform } from "@annotorious/annotorious";

  type PointEditorEvents = {
    change: Ellipse;
    grab: PointerEvent;
    release: PointerEvent;
  };

  export let shape: Ellipse;
  export let computedStyle: string | undefined;
  export let transform: Transform;
  export let viewportScale = 1;
  export let svgEl: SVGSVGElement | undefined;

  const dispatch = createEventDispatcher<PointEditorEvents>();
  const maskId = `point-editor-mask-${Math.random()
    .toString(36)
    .substring(2, 12)}`;

  let grabbedPointerId: number | null = null;
  let origin: [number, number] | null = null;
  let initialShape: Ellipse | null = null;
  let captureTarget: Element | null = null;

  $: geom = shape.geometry;
  $: scale = Math.max(Number(viewportScale) || 1, 0.0001);
  $: hitRx = Math.max(geom.rx, 8 / scale);
  $: hitRy = Math.max(geom.ry, 8 / scale);
  $: markerRadius = 2.4 / scale;
  $: maskPadding = 2 / scale;
  $: mask = {
    x: geom.bounds.minX - maskPadding,
    y: geom.bounds.minY - maskPadding,
    w: geom.bounds.maxX - geom.bounds.minX + 2 * maskPadding,
    h: geom.bounds.maxY - geom.bounds.minY + 2 * maskPadding,
  };

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
    initialShape = shape;
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
  <defs>
    <mask id={maskId} class="point-editor-mask">
      <rect x={mask.x} y={mask.y} width={mask.w} height={mask.h} />
      <ellipse cx={geom.cx} cy={geom.cy} rx={geom.rx} ry={geom.ry} />
    </mask>
  </defs>

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
    class="a9s-outer"
    mask={`url(#${maskId})`}
    on:pointerdown={grabShape}
    cx={geom.cx}
    cy={geom.cy}
    rx={geom.rx}
    ry={geom.ry}
  />

  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <ellipse
    class="a9s-inner a9s-shape-handle"
    style={computedStyle}
    on:pointerdown={grabShape}
    cx={geom.cx}
    cy={geom.cy}
    rx={geom.rx}
    ry={geom.ry}
  />

  <circle
    class="point-center-dot"
    cx={geom.cx}
    cy={geom.cy}
    r={markerRadius}
  />
</g>

<style>
  mask.point-editor-mask > rect {
    fill: #fff;
  }

  mask.point-editor-mask > ellipse {
    fill: #000;
  }

  .point-hit-target {
    fill: transparent;
    pointer-events: all;
    stroke: transparent;
    stroke-width: 0;
  }

  .point-center-dot {
    pointer-events: none;
    vector-effect: non-scaling-stroke;
  }

  .point-center-dot {
    fill: #fff;
    stroke: rgba(0, 0, 0, 0.9);
    stroke-width: 1.25px;
  }
</style>
