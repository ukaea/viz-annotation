"use client";
import { useEffect, useState } from "react";

/**
 * Tracks the live height of an element.
 *
 * Plotly needs an explicit pixel height, so a plot cannot simply stretch to fill
 * a flex parent. Attach the returned callback ref to the container the plot must
 * fill and feed the reported height into the plot layout.
 */
export function useElementHeight<T extends HTMLElement>() {
  // Held as state rather than a ref so the observer is attached as soon as the
  // element mounts - the container is often rendered after the first pass, once
  // the sample data has loaded.
  const [element, setElement] = useState<T | null>(null);
  const [height, setHeight] = useState(0);

  useEffect(() => {
    if (!element) return;

    setHeight(element.getBoundingClientRect().height);

    const observer = new ResizeObserver((entries) => {
      setHeight(entries[0].contentRect.height);
    });
    observer.observe(element);

    return () => observer.disconnect();
  }, [element]);

  return { ref: setElement, height };
}
