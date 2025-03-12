'use client'
import { useEffect, useRef, useState } from "react"
import { Menu, Item, Submenu, useContextMenu, ItemParams } from 'react-contexify'
import 'react-contexify/ReactContexify.css';
import * as d3 from "d3"

type GraphProps = {
    data: Array<{
        time: number,
        value: number 
    }>,
    shot_id: string
}

enum ZoneType {
    Type1,
    Type3
}

type ElmZone = {
    x0: number,
    x1: number,
    start: number,
    type: ZoneType
}

const MENU_ID = "zone_context"

export const ElmGraph = ({data, shot_id} : GraphProps) => {
    // SVG ref needed by D3 to update graph
    const svgRef = useRef(null)

    const spans = useRef<ElmZone[]>([])

    // Set up D3 scale bars - refs to allow useEffects to track and update them
    const width = 1300, height = 400, margin = 50

    const time_extent = d3.extent(data, d => d.time) as [number, number]
    const xScale = useRef(d3.scaleLinear().domain(time_extent).range([margin, width - margin]));
    const xScaleZoomedRef = useRef(xScale.current.copy()) // Copy required for zooming (investigate this)

    const value_extent = d3.extent(data, d => d.value) as [number, number]
    const yScale = useRef(d3.scaleLinear().domain(value_extent).range([height - margin, margin]));

    // Tracks the zones that have been added - can be monitored by zone useEffect
    const [updateZones, setUpdateZones] = useState(false)

    // Context menu set up including event callbacks
    const {show} = useContextMenu({
        id: MENU_ID
    })

    const handleDelete = ({props}: ItemParams) => {
        if (props.zone) {
            spans.current = spans.current.filter(span => span !== props.zone);
            setUpdateZones(update => !update);
        }
    };

    const handleTypeSetting = (target_zone: ElmZone, type: ZoneType) => {
        spans.current = spans.current.map((zone) => {
            if (zone === target_zone) {
                zone.type = type
            }
            return zone
        })
        setUpdateZones((update) => !update)
    }

    // Allows handles to resize zones - useEffect dependency handles redrawing
    const resizeDrag = d3.drag<Element, ElmZone>()
        .on("drag", function(event, d) {
            const handleType = d3.select(this).attr("data-handle");
            const newX = xScale.current.invert(event.x)

            if (handleType === "left") {
                d.x0 = Math.min(newX, d.x1 - 0.005);  // Prevent overlap (this should be revisted)
            } else {
                d.x1 = Math.max(newX, d.x0 + 0.005);
            }
            setUpdateZones((update) => !update)
        })

    // Handles the downloading of the zoning data
    const downloadData = () => {
        if (spans.current.length === 0) return;

        const csvContent = [
            "x0, x1, type",
            ...spans.current.map(zone => `${zone.x0},${zone.x1},${zone.type}`)
        ].join("\n");

        const blob = new Blob([csvContent], {type: "text/csv"});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a")

        a.href = url
        a.download = `zone_data_${shot_id}.csv`;
        document.body.appendChild(a)
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // Main graph rendering
    useEffect(() => {
        const svg = d3.select(svgRef.current)
            .attr("width", width)
            .attr("height", height)

        svg.selectAll("*").remove(); // Clear previous contents in case of re-draw

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
        const graphGroup = clipGroup.append("g")
            .attr("class", "graph-content")
               
        const xAxis = svg.append("g")
            .attr("transform", `translate(0, ${height - margin})`)
            .call(d3.axisBottom(xScale.current));

        const yAxis = svg.append("g")
            .attr("transform", `translate(${margin}, 0)`)
            .call(d3.axisLeft(yScale.current));

        const line = d3.line<{time: number; value: number}>()
            .x(d => xScale.current(d.time))
            .y(d => yScale.current(d.value));

        graphGroup.append("path")
            .datum(data)
            .attr("fill", "none")
            .attr("stroke", "blue")
            .attr("stroke-width", 2)
            .attr("d", line);

        // Handles the drawing of new zones
        const drag = d3.drag()
            .on("start", function (event) {
                const startX = xScaleZoomedRef.current.invert(event.x);
                spans.current.push({ x0: startX, x1: startX, start: startX, type: ZoneType.Type1 })
                setUpdateZones((update) => !update)
            })
            .on("drag", function (event) {
                const span = spans.current[spans.current.length - 1]
                const dragValue = xScaleZoomedRef.current.invert(event.x);
                if (dragValue < span.start) {
                    span.x0 = dragValue;
                    span.x1 = span.start;
                } else {
                    span.x0 = span.start;
                    span.x1 = dragValue;
                }
                setUpdateZones((update) => !update)
            })
            .on("end", function () {
                const span = spans.current[spans.current.length - 1]
                if (span.x0 > span.x1) [span.x0, span.x1] = [span.x1, span.x0];
                if (span.x1 - span.x0 < 0.01) {
                    spans.current.pop();
                }
                setUpdateZones((update) => !update)
            });

        // Handles zoom and panning
        const zoom = d3.zoom()
            .scaleExtent([1, 5])
            .translateExtent([[margin, margin], [width - margin, height - margin]])
            .on("zoom", (event) => {
                graphGroup.attr("transform", event.transform);
                xScaleZoomedRef.current = event.transform.rescaleX(xScale.current);
                xAxis.call(d3.axisBottom(xScaleZoomedRef.current));
                yAxis.call(d3.axisLeft(event.transform.rescaleY(yScale.current)));
                
                const scaleFactor = 1 / event.transform.k
                graphGroup.selectAll("path")
                    .attr("stroke-width", 2 * scaleFactor)
                    setUpdateZones((update) => !update)
            });

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

    }, [data])

    // Zone rendering
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
        const spanGroups = graphGroup.selectAll(".span-group").data(spans.current);

        const newGroups = spanGroups.enter()
            .append("g")
            .attr("class", "span-group");

        newGroups.append("rect")
            .attr("class", "span")
            .merge(spanGroups.select(".span"))
            .attr("x", d => xScale.current(d.x0))
            .attr("y", 0)
            .attr("width", d => xScale.current(d.x1) - xScale.current(d.x0))
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
            .call(resizeDrag)
    
        newGroups.append("circle")
            .attr("class", "handle right-handle")
            .attr("data-handle", "right")
            .attr("r", 5)
            .attr("fill", "red")
            .call(resizeDrag)

        spanGroups.select(".span")
            .attr("x", d => xScale.current(d.x0))
            .attr("width", d => xScale.current(d.x1) - xScale.current(d.x0));
        
        spanGroups.select(".left-handle")
            .attr("cx", d => xScale.current(d.x0))
            .attr("cy", height / 2);
        
        spanGroups.select(".right-handle")
            .attr("cx", d => xScale.current(d.x1))
            .attr("cy", height / 2);

        spanGroups.exit().remove();
    }, [resizeDrag, show, updateZones])

    return (
        <div style={{ display: "flex" }}>
            <div class="flex flex-col items-center space-y-3">
                <div class="w-full">
                    <svg ref={svgRef}></svg>
                </div>

                <div class='toolbar'>
                    <button class='btn-primary'
                        onClick={downloadData}
                    >Download Data</button>

                    <button class="btn-primary"
                        onClick={downloadData}
                    >Save</button>
                </div>

            </div>


            <Menu id={MENU_ID}>
                <Item id="delete" onClick={handleDelete}>Delete</Item>
                <Submenu label="Set type">
                    <Item id="zone1" onClick={({props}: ItemParams) => {
                        handleTypeSetting(props.zone, ZoneType.Type1)
                    }}>Zone I</Item>
                    <Item id="zone3" onClick={({props}: ItemParams) => {
                        handleTypeSetting(props.zone, ZoneType.Type3)
                    }}>Zone III</Item>
                </Submenu>
            </Menu>
        </div>
    )
}