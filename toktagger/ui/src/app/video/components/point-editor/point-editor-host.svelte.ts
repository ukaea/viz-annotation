import { mount, unmount } from "svelte";
import type { Ellipse, Transform } from "@annotorious/annotorious";

import PointEditor from "./PointEditor.svelte";

/** Props Annotorious pushes into a registered shape editor. */
export interface PointEditorProps {
  shape: Ellipse;
  computedStyle?: string;
  transform: Transform;
  viewportScale?: number;
  svgEl?: SVGSVGElement;
}

type PointEditorEventDetail = {
  change: Ellipse;
  grab: PointerEvent;
  release: PointerEvent;
};

type PointEditorEventName = keyof PointEditorEventDetail;
type PointEditorEventHandler = (event: {
  detail: PointEditorEventDetail[PointEditorEventName];
}) => void;

type PointEditorCallbacks = {
  onchange: (shape: Ellipse) => void;
  ongrab: (event: PointerEvent) => void;
  onrelease: (event: PointerEvent) => void;
};

type PointEditorComponentProps = PointEditorProps & PointEditorCallbacks;

interface PointEditorHostOptions {
  target: Element;
  props: PointEditorProps;
}

/**
 * Adapts Svelte 5's mount()/unmount() API to the Svelte 4-shaped editor
 * contract used by Annotorious: new / $set / $$set / $on / $destroy.
 */
export class PointEditorHost {
  #props: PointEditorComponentProps;
  #instance: ReturnType<typeof mount>;
  #handlers = new Map<PointEditorEventName, Set<PointEditorEventHandler>>();
  #destroyed = false;

  constructor(options: PointEditorHostOptions) {
    // This must be the first assignment to the field: $state makes subsequent
    // $set/$$set mutations visible to the mounted Svelte component.
    this.#props = $state({
      ...options.props,
      onchange: (shape) => this.#emit("change", shape),
      ongrab: (event) => this.#emit("grab", event),
      onrelease: (event) => this.#emit("release", event),
    });

    this.#instance = mount(PointEditor, {
      target: options.target,
      props: this.#props,
    });
  }

  #emit(
    event: PointEditorEventName,
    detail: PointEditorEventDetail[PointEditorEventName],
  ) {
    for (const handler of this.#handlers.get(event) ?? []) {
      handler({ detail });
    }
  }

  $set(props: Partial<PointEditorProps>) {
    if (this.#destroyed) return;
    Object.assign(this.#props, props);
  }

  // Annotorious calls $$set directly after every "change" event, echoing the
  // shape back into the editor. It must have the same semantics as $set.
  $$set(props: Partial<PointEditorProps>) {
    this.$set(props);
  }

  $on(event: PointEditorEventName, handler: PointEditorEventHandler) {
    const handlers =
      this.#handlers.get(event) ?? new Set<PointEditorEventHandler>();
    handlers.add(handler);
    this.#handlers.set(event, handlers);

    return () => {
      handlers.delete(handler);
    };
  }

  $destroy() {
    if (this.#destroyed) return;

    this.#destroyed = true;
    this.#handlers.clear();
    void unmount(this.#instance);
  }
}
