import {
  ShapeType,
  type AnnotoriousOpenSeadragonAnnotator,
} from "@annotorious/react";
import type { Ellipse, Transform } from "@annotorious/annotorious";
import type { SvelteComponent } from "svelte";
import { asClassComponent } from "svelte/legacy";

import PointEditor from "./PointEditor.svelte";

/** Props Annotorious pushes into a registered shape editor. */
interface PointEditorProps {
  shape: Ellipse;
  computedStyle?: string;
  transform: Transform;
  viewportScale: number;
  svgEl?: SVGSVGElement;
}

interface PointEditorComponentOptions {
  target: Element;
  props: PointEditorProps;
}

type PointEditorComponentInstance = SvelteComponent & {
  $set: (props: Partial<PointEditorProps>) => void;
  $$set: (props: Partial<PointEditorProps>) => void;
};

// Annotorious still instantiates shape editors with the Svelte 4 class API.
// Keep this legacy bridge until the adapter is rewritten with Svelte 5 mount().
const LegacyPointEditor = asClassComponent(PointEditor) as new (
  options: PointEditorComponentOptions,
) => PointEditorComponentInstance;

class PointEditorComponent extends LegacyPointEditor {
  constructor(options: PointEditorComponentOptions) {
    super(options);
    // Annotorious calls `$$set` directly after editor "change" events
    // (`u.$$set({ shape: d.detail })`). asClassComponent provides `$set`, but
    // not `$$set`; removing this breaks point dragging.
    this.$$set = (props: Partial<PointEditorProps>) => this.$set(props);
  }
}

export function registerPointEditor(
  api: Pick<AnnotoriousOpenSeadragonAnnotator, "registerShapeEditor">,
) {
  api.registerShapeEditor(
    ShapeType.ELLIPSE,
    // registerShapeEditor is typed for a Svelte 4 component constructor. This
    // adapter implements the required new/$set/$$set/$on/$destroy contract.
    PointEditorComponent as unknown as Parameters<
      typeof api.registerShapeEditor
    >[1],
  );
}
