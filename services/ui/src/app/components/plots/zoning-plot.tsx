import { useEffect, useState, useRef, JSX } from "react"

import * as Plotly from "plotly.js"
import * as d3 from "d3"

import { useZoneContext } from "../providers/zoning"
import { ZoneShape, Zone } from "../core/zone";

type PlotInfo = {
    data: Plotly.Data[];
    layout?: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
};


export const ZoningPlot = (info: PlotInfo) => {
    var { zones, categories, group } = useZoneContext()
    const mouseOffsetX = useRef<number>(0)
    const [entries, setEntries] = useState<JSX.Element[]>([])
    const [shouldRefresh, setShouldRefresh] = useState<boolean>(false)

    const init = (plot: HTMLElement) => {
        const overplot = plot.querySelector(".overplot")! as HTMLElement
        const xy = overplot.querySelector(".xy")! as HTMLElement

        var isFirstTime: boolean = true
        if (group.current) {
            isFirstTime = false
        }

        if (document.getElementsByClassName('d3zone_overplot').length > 0) {
            group.current = document.getElementsByClassName('d3zone_overplot')[0] as SVGGElement;
        } else {
            group.current = document.createElementNS("http://www.w3.org/2000/svg", "g")
            group.current.setAttribute("class", "d3zone_overplot");
            group.current.setAttribute("fill", "none");
            xy.appendChild(group.current)

            if (isFirstTime) {
                zones.current = []
                var offset = 0.05;
                for (const category of categories) {
                    zones.current.push({
                        category: category,
                        x0: offset,
                        x1: offset + 0.05,
                        isDragging: false,
                        isResizingLeft: false,
                        isResizingRight: false,
                        isVisible: true,
                        dragHandle: null,
                        resizeHandles: null
                    })
                    offset += 0.1
                }
            }
        }
    }

    const createZones = () => {
        const d3group = d3.select(group.current)
        d3group.selectAll("*").remove()

        var tableEntries = []

        for (var zone of zones.current) {
            tableEntries.push(tableEntry(zone))
            if (!zone.isVisible) {
                continue
            }
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

        setEntries(entries)
    }

    const updateZones = (plot: any) => {
        const xRange = plot._fullLayout.xaxis.range
        const xScale = plot._fullLayout.xaxis._length / (xRange[1] - xRange[0])

        const marginTop = plot._fullLayout._size.t
        const totalHeight = plot._fullLayout.yaxis._length

        const y = marginTop - totalHeight
        const height = 2 * totalHeight

        var tableEntries = []

        for (var zone of zones.current) {
            tableEntries.push(tableEntry(zone))
            if (!zone.isVisible) {
                continue
            }
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

        setEntries(tableEntries)
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
                const currentWidth = zone.x1 - zone.x0
                const newX0 = plot._fullLayout.xaxis.range[0] + (mouseX / xScale)
                zone.x0 = newX0
                zone.x1 = newX0 + currentWidth

                isActive = true;
            }
            if (zone.isResizingLeft) {
                const newX0 = plot._fullLayout.xaxis.range[0] + mouseX / xScale
                zone.x0 = newX0

                isActive = true;
            }
            if (zone.isResizingRight) {
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

    const toggleZoneVisibility = (zone: Zone) => {
        zone.isVisible = !zone.isVisible
        setShouldRefresh(true)
    }

    const tableEntry = (zone: Zone) => {
        const x0 = zone.x0.toFixed(6)
        const x1 = (zone.category.shape === ZoneShape.Rect) ? zone.x1.toFixed(6) : "--"
        const marker = (zone.category.shape === ZoneShape.Rect)
            ? "w-5 h-5 sm:rounded-lg"
            : "w-5 h-1.5 sm:rounded-lg"
        const opacity = (zone.isVisible)
            ? 1.0
            : 0.1
        return (
            <tr key={zone.category.name} className="bg-white border-b dark:bg-gray-800 dark:border-gray-700 border-gray-200 hover:bg-gray-100 dark:hover:bg-gray-600">
                <th scope="row" className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap dark:text-white">
                    <div className="flex space-x-3 items-center" style={{ cursor: "pointer", opacity: opacity }} onClick={(event) => toggleZoneVisibility(zone)}>
                        <div className={marker} style={{ background: zone.category.color }} />
                        <span>{zone.category.name}</span>
                    </div>
                </th>
                <td className="px-6 py-4">
                    <span style={{ opacity: opacity }}>{x0}</span>
                </td>
                <td className="px-6 py-4">
                    <span style={{ opacity: opacity }}>{x1}</span>
                </td>
            </tr >
        )
    }

    useEffect(() => {
        const root = document.getElementById("plot")!;

        Plotly.newPlot(root, info.data, info.layout, info.config).then((plot: Plotly.PlotlyHTMLElement) => {
            init(plot as HTMLElement)
            createZones()
            updateZones(plot)
            registerEventHandlers(plot)
            plot.on("plotly_relayout", () => {
                updateZones(plot)
            })
        })

        setShouldRefresh(false)

    }, [info, zones, shouldRefresh]);

    return (
        <div className="w-full px-6 py-3 space-y-3 flex-col">

            {/* Div where plot is inserted */}
            <div id="plot" className="" />

            {/* Table that works as legend to the plot */}
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

        </div>
    )
}
