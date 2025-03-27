import React from "react";

export enum ZoneShape {
    Line,
    Rect
}

export type ZoneCategory = {
    name: string;
    shape: ZoneShape;
    color: string;
}

export type Zone = {
    category: ZoneCategory;

    x0: number;
    x1: number;

    isDragging: boolean;
    isResizingLeft: boolean;
    isResizingRight: boolean;

    dragHandle: SVGRectElement | SVGLineElement | null;
    resizeHandles: SVGLineElement[] | null;
}