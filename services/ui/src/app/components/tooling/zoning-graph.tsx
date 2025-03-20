import React, { useEffect, useRef, createContext, useContext, useState } from "react"
import * as d3 from "d3"
import { useZoom } from "../providers/zoom-provider"
import { ELM_MENU_ID, ElmZone, useZones, ZoneType } from "../providers/zone-provider"
import { useContextMenu } from "react-contexify"
import 'react-contexify/ReactContexify.css';

interface GraphContextProps {
    xScale: d3.ScaleLinear<number, number, never>
    yScale: d3.ScaleLinear<number, number, never>
    graphGroup: SVGGElement | null
}
const GraphContext = createContext<GraphContextProps | null>(null)

type ZoningGraphProps = {
    yDomain: [number, number],
    height: number
    children: React.ReactNode
}

/**
 * This component provides a template for a graph that provides zone tooling
 * 
 * Additional subplots can be nested inside using useGraph.
 * @param Required parameters to generate axes and render subplots
 * @returns Main component with axes and parent SVG along with any nested subplots
 */
export const ZoningGraph = ({yDomain, height, children} : ZoningGraphProps) => {
    const svgRef = useRef<SVGSVGElement | null>(null)
    const [graphGroup, setGraphGroup] = useState<SVGGElement | null>(null)

    const {xScale, zoomTransform, handleZoom, sizing} = useZoom()
    const width = sizing.width, margin = sizing.margin

    const {handleZoneUpdate, zones} = useZones()

    const yScale = useRef(d3.scaleLinear().domain(yDomain).range([height - margin, margin]))

    // Context menu set up including event callbacks
    const {show} = useContextMenu({
        id: ELM_MENU_ID
    })

    // Allows handles to resize zones - useEffect dependency handles redrawing
    const resizeDrag = d3.drag<Element, ElmZone>()
        .on("drag", function(event, d) {
            const handleType = d3.select(this).attr("data-handle");
            const newX = xScale.invert(event.x)

            if (handleType === "left") {
                d.x0 = Math.min(newX, d.x1 - 0.005);  // Prevent overlap (this should be revisted)
            } else {
                d.x1 = Math.max(newX, d.x0 + 0.005);
            }
            handleZoneUpdate(null)
        })

    // Handles the creation of the graph content group
    useEffect(() => {
        const svg = d3.select(svgRef.current)
            .attr("width", width)
            .attr("height", height)
        
        // Clip path prevents rendering of graph outside of axes when scaled / panned
        svg.append("clipPath")
            .attr("id", "clip")
            .append("rect")
            .attr("x", margin)
            .attr("y", margin)
            .attr("width", width - 2 * margin)
            .attr("height", height - 2 * margin);

        const clipGroup = svg.append("g")
            .attr("clip-path", "url(#clip)")
        const g = clipGroup.append("g")
            .attr("class", "graph-content")
        setGraphGroup(g.node())
    }, [height, margin, width])

    // Handles axis rendering
    useEffect(() => {
        const svg = d3.select(svgRef.current)

        svg.selectAll(".axis").remove(); // Clear previous contents in case of re-draw
                
        svg.append("g")
            .attr("class", "axis")
            .attr("transform", `translate(0, ${height - margin})`)
            .call(d3.axisBottom(xScale));

        svg.append("g")
            .attr("class", "axis")
            .attr("transform", `translate(${margin}, 0)`)
            .call(d3.axisLeft(yScale.current));
    }, [height, margin, xScale])

    // Handles D3 event handlers for zone creation and zoom/pan
    useEffect(() => {
        const svg = d3.select(svgRef.current)
        const g = svg.select(".graph-content")

        // Zone creation handler
        const drag = d3.drag()
            .on("start", function (event) {
                const startX = xScale.invert(event.x);
                zones.push({ x0: startX, x1: startX, start: startX, type: ZoneType.Type1 })
                handleZoneUpdate(null)
            })
            .on("drag", function (event) {
                const zone = zones[zones.length - 1]
                const dragValue = xScale.invert(event.x)
                if (dragValue < zone.start) {
                    zone.x0 = dragValue;
                    zone.x1 = zone.start;
                } else {
                    zone.x0 = zone.start;
                    zone.x1 = dragValue;
                }
                handleZoneUpdate(null)
            })
            .on("end", function () {
                const zone = zones[zones.length - 1]
                if (zone.x0 > zone.x1) [zone.x0, zone.x1] = [zone.x1, zone.x0];
                if (zone.x1 - zone.x0 < 0.01) {
                    zones.pop();
                }
                handleZoneUpdate(null)
            });

        // Zoom handler
        const zoom = d3.zoom()
            .scaleExtent([1, 5])
            .translateExtent([[margin, margin], [width - margin, height - margin]])
            .on("zoom", (event) => {
                handleZoom(event)
                
                const scaleFactor = 1 / event.transform.k
                g.selectAll("path")
                    .attr("stroke-width", 2 * scaleFactor)
                handleZoneUpdate(null)
            });
        
        // This is required to keep linked axes synchronised
        if (d3.zoomTransform(svg.node()).toString() !== zoomTransform.toString()) {
            svg.call(zoom.transform, zoomTransform)
        }

        // Initially set up the graph with the zone creation functionality
        let isShiftPressed = false;
        svg.call(drag) // Typing of these calls should be visted

        // If the shift key is pressed change to panning functionality
        const keydownHandler = (event: KeyboardEvent) => {
            if (event.key === "Shift" && !isShiftPressed) {
                isShiftPressed = true;
                svg.on(".drag", null);
                svg.call(zoom);
            }
        };

        // Return to zoning after shift is released
        const keyupHandler = (event: KeyboardEvent) => {
            if (event.key === "Shift") {
                isShiftPressed = false;
                svg.on(".zoom", null);
                svg.call(drag);
            }
        };

        document.addEventListener("keydown", keydownHandler);
        document.addEventListener("keyup", keyupHandler);

        // Ensure listeners are cleaned up correctly
        return () => {
            document.removeEventListener("keydown", keydownHandler);
            document.removeEventListener("keyup", keyupHandler);
        };
    }, [handleZoneUpdate, handleZoom, height, margin, width, xScale, zones, zoomTransform])

    useEffect(() => {
        // Required to allow context menu to be triggered
        function handleContextMenu(event, zone: ElmZone) {
            show({
                event,
                props: {
                    key: 'value',
                    zone,
                }
            })
        }

        // Pulls the graph group from the SVG to ensure zones are correctly drawn
        const graphGroup = d3.select(svgRef.current).select(".graph-content")

        // Data binding to efficiently handles creation / update / deletion of zones
        const spanGroups = graphGroup.selectAll(".span-group").data(zones);

        const newGroups = spanGroups.enter()
            .append("g")
            .attr("class", "span-group");

        newGroups.append("rect")
            .attr("class", "span")
            .merge(spanGroups.select(".span"))
            .attr("x", d => xScale(d.x0))
            .attr("y", 0)
            .attr("width", d => xScale(d.x1) - xScale(d.x0))
            .attr("height", height)
            .attr("fill", d => {
                if (d.type == ZoneType.Type1) return "rgba(0, 0, 255, 0.3)";
                if (d.type == ZoneType.Type3) return "rgba(0, 255, 0, 0.3)";
                return "rgba(255, 0, 0, 0.3)";
            })
            .on("contextmenu", handleContextMenu); // Assigns context menu callback to each zone

        newGroups.append("circle")
            .attr("class", "handle left-handle")
            .attr("data-handle", "left")
            .attr("r", 5)
            .attr("fill", "red")
            .attr("cx", d => xScale(d.x0))
            .attr("cy", height / 2)
            .call(resizeDrag)
    
        newGroups.append("circle")
            .attr("class", "handle right-handle")
            .attr("data-handle", "right")
            .attr("r", 5)
            .attr("fill", "red")
            .attr("cx", d => xScale(d.x1))
            .attr("cy", height / 2)
            .call(resizeDrag)

        spanGroups.select(".span")
            .attr("x", d => xScale(d.x0))
            .attr("width", d => xScale(d.x1) - xScale(d.x0));
        
        spanGroups.select(".left-handle")
            .attr("cx", d => xScale(d.x0))
            .attr("cy", height / 2);
        
        spanGroups.select(".right-handle")
            .attr("cx", d => xScale(d.x1))
            .attr("cy", height / 2);

        spanGroups.exit().remove();
    }, [height, resizeDrag, show, xScale, zones])

    return (
        <GraphContext.Provider value={{xScale, yScale: yScale.current, graphGroup}}>
            <svg ref={svgRef}>{children}</svg>
        </GraphContext.Provider>
    )
}

/**
 * Allows sub plots access to scaling data and the parent SVG - prevents use outside of provider
 * @returns context for destructuring
 */
export const useGraph = () => {
    const context = useContext(GraphContext);
    if (!context) {
        throw new Error("useGraph must be used within a GraphProvider");
    }
    return context;
};