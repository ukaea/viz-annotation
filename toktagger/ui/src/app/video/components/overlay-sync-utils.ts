import { boundsFromPoints, type ImageAnnotation } from "@annotorious/react";
import {
  getLabelTrack,
  isEllipseAnno,
  isPointAnno,
  isRectangleAnno,
  isPolygonAnno,
  POINT_MARKER_SIZE,
  type EllipseAnnotation,
  type PolygonAnnotation,
  type RectangleAnnotation,
  readEllipseGeometry,
  readPointGeometry,
  readPolygonGeometry,
  readRectGeometry,
  VideoImageAnnotation,
} from "./anno-utils";

export type ImagePoint = { x: number; y: number };

function getTargetSource(a: ImageAnnotation): string {
  return (a as VideoImageAnnotation).target.source ?? "";
}

function polygonContainsPoint(points: [number, number][], point: ImagePoint) {
  let inside = false;

  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i];
    const [xj, yj] = points[j];
    const intersects =
      yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi;

    if (intersects) inside = !inside;
  }

  return inside;
}

export function annotationContainsPoint(
  annotation: ImageAnnotation,
  point: ImagePoint,
) {
  const ellipse = isEllipseAnno(annotation)
    ? readEllipseGeometry(annotation)
    : null;
  if (ellipse) {
    const dx = (point.x - ellipse.cx) / ellipse.rx;
    const dy = (point.y - ellipse.cy) / ellipse.ry;
    return dx * dx + dy * dy <= 1;
  }

  const rect = isRectangleAnno(annotation)
    ? readRectGeometry(annotation)
    : null;
  if (rect) {
    return (
      point.x >= rect.x &&
      point.x <= rect.x + rect.w &&
      point.y >= rect.y &&
      point.y <= rect.y + rect.h
    );
  }

  const polygon = isPolygonAnno(annotation)
    ? readPolygonGeometry(annotation)
    : null;
  if (!polygon) return false;

  const { minX, minY, maxX, maxY } = polygon.bounds;
  if (point.x < minX || point.x > maxX || point.y < minY || point.y > maxY) {
    return false;
  }

  return polygonContainsPoint(polygon.points, point);
}

export function setViewerCursor(viewerElement: HTMLElement, cursor: string) {
  const applyCursor = (target: HTMLElement | SVGElement) => {
    if (cursor) {
      target.style.setProperty("cursor", cursor, "important");
    } else {
      target.style.removeProperty("cursor");
    }
  };

  const scopes = [
    viewerElement,
    viewerElement.parentElement,
    viewerElement.parentElement?.parentElement,
  ].filter(Boolean) as HTMLElement[];

  applyCursor(viewerElement);

  const selector = [
    ".a9s-gl-canvas",
    ".a9s-annotationlayer",
    ".a9s-annotationlayer *",
    ".a9s-annotation",
    ".a9s-annotation *",
  ].join(", ");

  scopes.forEach((scope) => {
    scope
      .querySelectorAll<HTMLElement | SVGElement>(selector)
      .forEach(applyCursor);
  });
}

/**
 * Stable signature for comparing overlays by content (not by reference).
 * Used to avoid feedback loops when we programmatically call `setAnnotations`.
 */
function annoSig(a: ImageAnnotation): string {
  const sel = a.target.selector;
  const source = getTargetSource(a);
  const rect = isRectangleAnno(a) ? readRectGeometry(a) : null;
  const ellipse = isEllipseAnno(a) ? readEllipseGeometry(a) : null;
  const poly = isPolygonAnno(a) ? readPolygonGeometry(a) : null;
  const point = isPointAnno(a) ? readPointGeometry(a) : null;
  const { className, trackId } = getLabelTrack(a);
  const polySig = poly
    ? poly.points.map(([x, y]) => `${x},${y}`).join(";")
    : "";
  const pointSig = point ? `${point.x},${point.y}` : "";

  return [
    a.id ?? "",
    sel?.type ?? "",
    source,
    isPointAnno(a) ? "point" : "",
    rect?.x ?? "",
    rect?.y ?? "",
    rect?.w ?? "",
    rect?.h ?? "",
    ellipse?.cx ?? "",
    ellipse?.cy ?? "",
    ellipse?.rx ?? "",
    ellipse?.ry ?? "",
    polySig,
    pointSig,
    className ?? "",
    trackId ?? "",
  ].join("|");
}

export function sameOverlay(
  a: ImageAnnotation[],
  b: ImageAnnotation[],
): boolean {
  if (a === b) return true;
  if (!Array.isArray(a) || !Array.isArray(b)) return false;
  if (a.length !== b.length) return false;

  const as = a.map(annoSig).sort();
  const bs = b.map(annoSig).sort();
  for (let i = 0; i < as.length; i++) if (as[i] !== bs[i]) return false;
  return true;
}

export function clampRectToImage(
  g: { x: number; y: number; w: number; h: number },
  nw: number,
  nh: number,
): { x: number; y: number; w: number; h: number } | null {
  const x1 = Math.min(g.x, g.x + g.w);
  const y1 = Math.min(g.y, g.y + g.h);
  const x2 = Math.max(g.x, g.x + g.w);
  const y2 = Math.max(g.y, g.y + g.h);

  const cx1 = Math.max(0, Math.min(nw, x1));
  const cy1 = Math.max(0, Math.min(nh, y1));
  const cx2 = Math.max(0, Math.min(nw, x2));
  const cy2 = Math.max(0, Math.min(nh, y2));

  const w = cx2 - cx1;
  const h = cy2 - cy1;

  if (!(w > 0 && h > 0)) return null;
  return { x: cx1, y: cy1, w, h };
}

function withRectGeometry(
  a: RectangleAnnotation,
  g: { x: number; y: number; w: number; h: number },
): RectangleAnnotation {
  const nextGeom = {
    ...a.target.selector.geometry,
    x: g.x,
    y: g.y,
    w: g.w,
    h: g.h,
    bounds: { minX: g.x, minY: g.y, maxX: g.x + g.w, maxY: g.y + g.h },
  };

  return {
    ...a,
    target: {
      ...a.target,
      selector: {
        ...a.target.selector,
        geometry: nextGeom,
      },
    },
  };
}

function withEllipseGeometry(
  a: EllipseAnnotation,
  g: { cx: number; cy: number; rx: number; ry: number },
): EllipseAnnotation {
  const nextGeom = {
    ...a.target.selector.geometry,
    cx: g.cx,
    cy: g.cy,
    rx: g.rx,
    ry: g.ry,
    bounds: {
      minX: g.cx - g.rx,
      minY: g.cy - g.ry,
      maxX: g.cx + g.rx,
      maxY: g.cy + g.ry,
    },
  };

  return {
    ...a,
    target: {
      ...a.target,
      selector: {
        ...a.target.selector,
        geometry: nextGeom,
      },
    },
  };
}

function clampPoint(
  point: ReadonlyArray<number>,
  nw: number,
  nh: number,
): [number, number] {
  const [x, y] = point;
  return [Math.max(0, Math.min(nw, x)), Math.max(0, Math.min(nh, y))];
}

function samePoint(
  a: ReadonlyArray<number>,
  b: ReadonlyArray<number>,
): boolean {
  return a[0] === b[0] && a[1] === b[1];
}

function polygonArea(points: Array<ReadonlyArray<number>>): number {
  let area = 0;

  for (let i = 0; i < points.length; i += 1) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    area += x1 * y2 - x2 * y1;
  }

  return Math.abs(area) / 2;
}

function clampPolygonToImage(
  points: Array<Array<number>>,
  nw: number,
  nh: number,
): [number, number][] | null {
  const clamped = points.map((point) => clampPoint(point, nw, nh));
  const deduped: [number, number][] = [];

  for (const point of clamped) {
    if (
      deduped.length === 0 ||
      !samePoint(deduped[deduped.length - 1], point)
    ) {
      deduped.push(point);
    }
  }

  if (
    deduped.length > 1 &&
    samePoint(deduped[0], deduped[deduped.length - 1])
  ) {
    deduped.pop();
  }

  const unique = new Set(deduped.map(([x, y]) => `${x},${y}`));
  if (unique.size < 3) return null;
  if (polygonArea(deduped) <= 0) return null;

  return deduped;
}

function withPolygonGeometry(
  a: PolygonAnnotation,
  points: [number, number][],
): PolygonAnnotation {
  const nextGeom = {
    ...a.target.selector.geometry,
    points,
    bounds: boundsFromPoints(points),
  };

  return {
    ...a,
    target: {
      ...a.target,
      selector: {
        ...a.target.selector,
        geometry: nextGeom,
      },
    },
  };
}

export function clampOverlayToNaturalImage(
  list: ImageAnnotation[],
  natural: { w: number; h: number } | null,
): ImageAnnotation[] {
  if (!natural?.w || !natural?.h) return list;

  let changed = false;
  const out: ImageAnnotation[] = [];

  for (const a of list) {
    if (isPointAnno(a)) {
      const point = readPointGeometry(a);
      if (!point) {
        out.push(a);
        continue;
      }

      const x = Math.max(0, Math.min(natural.w, point.x));
      const y = Math.max(0, Math.min(natural.h, point.y));

      if (x === point.x && y === point.y) out.push(a);
      else {
        changed = true;
        const ellipse = readEllipseGeometry(a);
        const rx = ellipse?.rx ?? POINT_MARKER_SIZE / 2;
        const ry = ellipse?.ry ?? POINT_MARKER_SIZE / 2;
        out.push(
          withEllipseGeometry(a, {
            cx: x,
            cy: y,
            rx,
            ry,
          }),
        );
      }
      continue;
    }

    if (isRectangleAnno(a)) {
      const g = readRectGeometry(a);
      if (!g) {
        out.push(a);
        continue;
      }

      const clamped = clampRectToImage(g, natural.w, natural.h);
      if (!clamped) {
        changed = true;
        continue;
      }

      const same =
        clamped.x === g.x &&
        clamped.y === g.y &&
        clamped.w === g.w &&
        clamped.h === g.h;

      if (same) out.push(a);
      else {
        changed = true;
        out.push(withRectGeometry(a, clamped));
      }
      continue;
    }

    if (!isPolygonAnno(a)) {
      out.push(a);
      continue;
    }

    const polygon = readPolygonGeometry(a);
    if (!polygon) {
      out.push(a);
      continue;
    }

    const clamped = clampPolygonToImage(polygon.points, natural.w, natural.h);
    if (!clamped) {
      changed = true;
      continue;
    }

    const same =
      clamped.length === polygon.points.length &&
      clamped.every(
        ([x, y], index) =>
          x === polygon.points[index]?.[0] && y === polygon.points[index]?.[1],
      );

    if (same) out.push(a);
    else {
      changed = true;
      out.push(withPolygonGeometry(a, clamped));
    }
  }

  return changed ? out : list;
}
