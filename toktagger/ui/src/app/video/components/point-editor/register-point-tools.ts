import {
  ShapeType,
  type AnnotoriousOpenSeadragonAnnotator,
} from "@annotorious/react";

import { PointEditorHost } from "./point-editor-host.svelte.ts";

export function registerPointEditor(
  api: Pick<AnnotoriousOpenSeadragonAnnotator, "registerShapeEditor">,
) {
  api.registerShapeEditor(
    ShapeType.ELLIPSE,
    // registerShapeEditor is typed for a Svelte 4 component constructor. This
    // Svelte 5 adapter implements the required new/$set/$$set/$on/$destroy
    // contract without relying on Svelte's deprecated legacy bridge.
    PointEditorHost as unknown as Parameters<typeof api.registerShapeEditor>[1],
  );
}
