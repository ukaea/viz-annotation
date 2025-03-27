import React, { useContext, createContext, useRef } from "react";

import * as d3 from "d3";

import { Zone, ZoneCategory, ZoneShape } from "../core/zone";

interface ZoneContextInfo {
    zones: Zone[]
    categories: ZoneCategory[]

    group: SVGGElement | null

    init: (plot: HTMLElement) => void
    createZones: () => void
    updateZones: (plot: any) => void
    registerEventHandlers: (plot: any) => void
}

const ZoneContext = createContext<ZoneContextInfo | null>(null)

export const useZoneContext = () => {
    const context = useContext(ZoneContext)
    if (!context) {
        throw new Error("useZone must be used within a ZoneProvider")
    }
    return context
}

export const ZoneProvider = ({ categories, children }: { categories: ZoneCategory[], children: React.ReactNode }) => {
    const zones = useRef<Zone[]>([])
    const group = useRef<SVGGElement | null>(null)
    const cat = useRef<ZoneCategory[]>(categories)
    const mouseOffsetX = useRef<number>(0)

    const init = (plot: HTMLElement) => {
        const overplot = plot.querySelector(".overplot")! as HTMLElement
        const xy = overplot.querySelector(".xy")! as HTMLElement

        if (!group.current) {
            group.current = document.createElementNS("http://www.w3.org/2000/svg", "g")
            group.current.setAttribute("class", "zones");
            group.current.setAttribute("fill", "none");
            xy.appendChild(group.current)

            zones.current = []
            var offset = 0.1;
            for (const category of cat.current) {
                zones.current.push({
                    category: category,
                    x0: offset,
                    x1: offset + 0.1,
                    isDragging: false,
                    isResizingLeft: false,
                    isResizingRight: false,
                    dragHandle: null,
                    resizeHandles: null
                })
                offset += 0.15
            }
        }
    }

    const createZones = () => {
        const d3group = d3.select(group.current)
        d3group.selectAll("*").remove()

        for (var zone of zones.current) {
            if (zone.category.shape === ZoneShape.Rect) {
                const rect = d3group.append("rect").attr("id", "zone-" + zone.category.name + "-rect")
                const left = d3group.append("line").attr("id", "zone-" + zone.category.name + "-left")
                const right = d3group.append("line").attr("id", "zone-" + zone.category.name + "-right")
                zone.dragHandle = rect.node() as SVGRectElement
                zone.resizeHandles = [left.node() as SVGLineElement, right.node() as SVGLineElement]
                zone.dragHandle.style.pointerEvents = "all"
                zone.resizeHandles[0].style.pointerEvents = "all"
                zone.resizeHandles[1].style.pointerEvents = "all"
            } else if (zone.category.shape === ZoneShape.Line) {
                const line = d3group.append("line").attr("id", "zone-" + zone.category.name + "-line")
                zone.dragHandle = line.node() as SVGLineElement
                zone.dragHandle.style.pointerEvents = "all"
            }
        }
    }

    const updateZones = (plot: any) => {
        const xRange = plot._fullLayout.xaxis.range
        const xScale = plot._fullLayout.xaxis._length / (xRange[1] - xRange[0])

        const marginTop = plot._fullLayout._size.t
        const totalHeight = plot._fullLayout.yaxis._length

        const y = marginTop - totalHeight
        const height = 2 * totalHeight

        for (var zone of zones.current) {
            if (zone.category.shape === ZoneShape.Rect) {
                var rect = d3.select(zone.dragHandle!)
                var left = d3.select(zone.resizeHandles![0])
                var right = d3.select(zone.resizeHandles![1])

                const x0px = xScale * (zone.x0 - xRange[0])
                const widthPx = xScale * (zone.x1 - zone.x0)

                rect
                    .attr("x", x0px)
                    .attr("y", y)
                    .attr("width", widthPx)
                    .attr("height", height)
                    .attr("fill", zone.category.color)
                    .attr("opacity", 0.1)
                    .style("cursor", "move")

                left
                    .attr("x1", x0px)
                    .attr("x2", x0px)
                    .attr("y1", y)
                    .attr("y2", y + height)
                    .attr("stroke", zone.category.color)
                    .attr("stroke-width", 3)
                    .style("cursor", "ew-resize")


                right
                    .attr("x1", x0px + widthPx)
                    .attr("x2", x0px + widthPx)
                    .attr("y1", y)
                    .attr("y2", y + height)
                    .attr("stroke", zone.category.color)
                    .attr("stroke-width", 3)
                    .style("cursor", "ew-resize")
            } else if (zone.category.shape === ZoneShape.Line) {
                var line = d3.select(zone.dragHandle!)

                const x0px = xScale * (zone.x0 - xRange[0])

                line
                    .attr("x1", x0px)
                    .attr("x2", x0px)
                    .attr("y1", y)
                    .attr("y2", y + height)
                    .attr("stroke", zone.category.color)
                    .attr("stroke-width", 6)
                    .style("cursor", "ew-resize")
            }
        }
    }

    const mouseDownHandler = (event: MouseEvent) => {
        for (var zone of zones.current) {
            var isActive = false

            const target = (event.target as Element).id

            if (zone.category.shape === ZoneShape.Rect) {
                if (target === zone.dragHandle!.id) {
                    zone.isDragging = true
                    mouseOffsetX.current = event.clientX - parseFloat(zone.dragHandle!.getAttribute("x")!);
                    isActive = true
                } else if (target === zone.resizeHandles![0].id) {
                    zone.isResizingLeft = true
                    mouseOffsetX.current = event.clientX - parseFloat(zone.resizeHandles![0].getAttribute("x1")!)
                    isActive = true
                } else if (target === zone.resizeHandles![1].id) {
                    zone.isResizingRight = true
                    mouseOffsetX.current = event.clientX - parseFloat(zone.resizeHandles![1].getAttribute("x1")!)
                    isActive = true
                }
            } else if (zone.category.shape == ZoneShape.Line) {
                if (target === zone.dragHandle!.id) {
                    zone.isDragging = true
                    mouseOffsetX.current = event.clientX - parseFloat(zone.dragHandle!.getAttribute("x1")!);
                    isActive = true
                }
            }

            if (isActive) {
                return
            }
        }
    }

    const mouseMoveHandler = (event: MouseEvent, plot: any) => {
        const xScale = plot._fullLayout.xaxis._length / (plot._fullLayout.xaxis.range[1] - plot._fullLayout.xaxis.range[0])
        const mouseX = event.clientX - mouseOffsetX.current

        for (var zone of zones.current) {
            var isActive = false
            if (zone.isDragging) {
                console.log(zone.category.name + " is dragging")
                const currentWidth = zone.x1 - zone.x0
                const newX0 = plot._fullLayout.xaxis.range[0] + (mouseX / xScale)
                zone.x0 = newX0
                zone.x1 = newX0 + currentWidth

                isActive = true;
            }
            if (zone.isResizingLeft) {
                console.log(zone.category.name + " is resizing left")
                const newX0 = plot._fullLayout.xaxis.range[0] + mouseX / xScale
                zone.x0 = newX0

                isActive = true;
            }
            if (zone.isResizingRight) {
                console.log(zone.category.name + " is resizing right")
                const newX1 = plot._fullLayout.xaxis.range[0] + mouseX / xScale
                zone.x1 = newX1

                isActive = true;
            }

            if (isActive) {
                updateZones(plot);
                document.removeEventListener("mousemove", (event) => mouseMoveHandler(event, plot))
                return
            }
        }

    }

    const mouseUpHandler = () => {
        for (var zone of zones.current) {
            if (zone.category.shape === ZoneShape.Rect) {
                zone.isDragging = false
                zone.isResizingLeft = false
                zone.isResizingRight = false
            } else if (zone.category.shape === ZoneShape.Line) {
                zone.isDragging = false
            }
        }
    }

    const registerEventHandlers = (plot: any) => {
        document.addEventListener("mousedown", mouseDownHandler)
        document.addEventListener("mousemove", (event) => mouseMoveHandler(event, plot))
        document.addEventListener("mouseup", mouseUpHandler)
    }


    return (
        <ZoneContext.Provider value={{
            zones: zones.current,
            categories: cat.current,
            group: group.current,
            init,
            createZones,
            updateZones,
            registerEventHandlers
        }}>
            {children}
        </ZoneContext.Provider>
    )
}
