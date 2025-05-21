import { useEffect, useRef } from "react"
import { useContextMenu } from "react-contexify";

import * as d3 from "d3"
import { useZoneContext, ZONE_MENU_ID } from "../providers/zone-provider";
import { Zone } from "@/types";

type ZoneProps = {
    plotId: string;
    plotReady: boolean;
    forceUpdate: number;
}

/**
 * Handles the rendering of zones onto a specific plot
 * 
 * @param plotId Used to identify the plot that the tooling should be rendered on
 * @param plotReady Signal from main plot that tooling can be drawn
 */
export const Zones = ({
    plotId,  
    plotReady, 
    forceUpdate
} : ZoneProps) => {
    const dragOffset = useRef(0)

    // Hook to trigger the context provider to render context menu
    const {show: showZoneMenu} = useContextMenu({
        id: ZONE_MENU_ID
    })

    // Hook to pull in data from context provider
    const {zones, handleZoneUpdate, triggerUpdate} = useZoneContext()

    // Main rendering effect
    useEffect(() => {
        // This shall not run until the target plot is initialised
        if (!plotReady) {
            return
        }

        // Grab the handle set up in the main plot for D3 rendering
        const plot = document.getElementById(plotId)

        // Rendering should not be attempted if the required handles are not found
        if (!plot) {
            console.error("Could not locate plot to generate zones")
            return
        }

        // Get a reference to all subplots and find the name of the axis
        const subplots = plot.querySelectorAll(".subplot")
        const subplotNames = [...subplots].map(el => 
        [...el.classList].find(cls => cls !== "subplot")
        )

        // For each subplot carry out the tooling generation
        subplotNames.forEach((subplotId => {
            if (subplotId === undefined) {
                console.error("Could not find valid subplot ID")
                return
            }

            const overplot = document.getElementsByClassName(`${plotId}-overplot-${subplotId}`)[0]
            
            if (!overplot) {
                console.error("Could not locate D3 overplot to generate zones")
                handleZoneUpdate()
                return
            }

            // Find the y axis ID relating to this subplot
            const yAxisID = subplotId.match(/y(.*)$/)?.[1];
            if (!yAxisID && yAxisID !== "") {
                console.error("Could not find valid subplot y-axis ID")
                return
            }
            // Use the axis information to calculate the upper and lower limits of the zone
            const axis = plot._fullLayout[`yaxis${yAxisID}`]
            const limits = axis.range
            const range = limits[1]-limits[0]
            const upperLimit = axis.d2p(limits[1] + 2*range)
            const lowerLimit =  axis.d2p(limits[0] - 2*range)
            const height = lowerLimit - upperLimit

            const xaxis = plot._fullLayout.xaxis;

            const graphGroup = d3.select(overplot)
            graphGroup.selectAll(".zone").remove() // All zones are removed each render cycle

            // Prevents a little bit of repetition by auto-configuring the resize handler
            const getBoundaryHandler = (isLeft: boolean) => {
                // Handles the dragging of the boundaries of the zone
                const resize = d3.drag<SVGRectElement, Zone>()
                    .on("drag", function (event, d) {
                        const x = xaxis.p2d(event.x);
                        if (isLeft) {
                            d.x0 = x
                        } else {
                            d.x1 = x
                        }
                        handleZoneUpdate()
                    })
                return resize
            }

            // Handles the dragging of the zones itself
            const drag = d3.drag<SVGRectElement, Zone>()
                .on("start", function (event, d) {
                    dragOffset.current = xaxis.d2p(d.x0) - event.x
                })
                .on("drag", function (event, d) {
                    const newX = event.x + dragOffset.current;
                    d3.select(this).attr("x", newX);

                    const x0 = xaxis.p2d(newX);
                    const x1 = xaxis.p2d(newX + xaxis.d2p(d.x1) - xaxis.d2p(d.x0));
                    d.x0 = x0;
                    d.x1 = x1;
                    handleZoneUpdate()
                })

            function handleContextMenu(event, zone: Zone) {
                showZoneMenu({
                    event,
                    props: {
                        zone
                    }
                })
            }

            // Create the zone and transparent handles on each boundary
            for (const zone of zones) {
                const x0 = xaxis.d2p(zone.x0);
                const x1 = xaxis.d2p(zone.x1);
                graphGroup.append("rect")
                    .attr("class", "zone span cursor-grab")
                    .attr("x", x0)
                    .attr("y", upperLimit)
                    .attr("width", x1 - x0)
                    .attr("height", height)
                    .attr("fill", zone.category.color)
                    .attr("opacity", 0.3)
                    .attr("style", "pointer-events: all")
                    .style("cursor", "move")
                    .datum(zone)
                    .call(drag)
                    .on("contextmenu", handleContextMenu)

                graphGroup.append("rect")
                    .attr("class", "zone leftHandle")
                    .attr("x", x0-10)
                    .attr("y", upperLimit)
                    .attr("width", 20)
                    .attr("height", height)
                    .attr("fill", "transparent")
                    .attr("style", "pointer-events: all")
                    .style("cursor", "move")
                    .datum(zone)
                    .call(getBoundaryHandler(true))

                graphGroup.append("rect")
                    .attr("class", "zone rightHandle")
                    .attr("x", x1-10)
                    .attr("y", upperLimit)
                    .attr("width", 20)
                    .attr("height", height)
                    .attr("fill", "transparent")
                    .attr("style", "pointer-events: all")
                    .style("cursor", "move")
                    .datum(zone)
                    .call(getBoundaryHandler(false))
            }
        }))
        
    }, [handleZoneUpdate, plotId, plotReady, showZoneMenu, zones, triggerUpdate, forceUpdate])

    return (
        <div />
    )
}