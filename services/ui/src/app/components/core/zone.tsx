import React, { useEffect, useId } from "react";

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

function rgbToHex(rgb: string): string {
    const match = rgb.match(/^rgb\((\d+),\s*(\d+),\s*(\d+)\)$/);
    if (!match) {
        throw new Error("Invalid RGB format. Expected format: 'rgb(r, g, b)'");
    }

    const r = parseInt(match[1]).toString(16).padStart(2, "0");
    const g = parseInt(match[2]).toString(16).padStart(2, "0");
    const b = parseInt(match[3]).toString(16).padStart(2, "0");

    return `#${r}${g}${b}`;
}

const ZoneTableEntry = (zone: Zone) => {
    const x0 = zone.x0.toFixed(6)
    const x1 = (zone.category.shape === ZoneShape.Rect) ? zone.x1.toFixed(6) : "--"
    const marker = (zone.category.shape === ZoneShape.Rect)
        ? "w-5 h-5 sm:rounded-lg"
        : "w-5 h-1.5 sm:rounded-lg"
    console.log(marker)
    return (
        <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700 border-gray-200 hover:bg-gray-100 dark:hover:bg-gray-600">
            <th scope="row" className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap dark:text-white">
                <div className="flex space-x-3 items-center">
                    <div className={marker} style={{ background: zone.category.color }} />
                    <span>{zone.category.name}</span>
                </div>
            </th>
            <td className="px-6 py-4">
                {x0}
            </td>
            <td className="px-6 py-4">
                {x1}
            </td>
        </tr >
    )
}

export const ZoneTable = ({ zones }: { zones: Zone[] }) => {

    const entries: any = []

    for (const zone of zones) {
        entries.push(ZoneTableEntry(zone))
    }

    return (
        <div className="relative w-fit overflow-x-auto shadow-md sm:rounded-lg ml-auto mr-auto">
            <table className="w-full text-sm text-left rtl:text-right text-gray-500 dark:text-gray-400">
                <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
                    <tr>
                        <th scope="col" className="px-6 py-3">
                            Zone category
                        </th>
                        <th scope="col" className="px-6 py-3">
                            x0
                        </th>
                        <th scope="col" className="px-6 py-3">
                            x1
                        </th>
                    </tr>
                </thead>
                <tbody>
                    {entries}
                </tbody>
            </table>
        </div>

    )
}