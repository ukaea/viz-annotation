import { useEffect, useRef } from "react"
import { useContextMenu } from "react-contexify";

import * as d3 from "d3"
import { useVSpanContext, VSPAN_MENU_ID } from "../providers/vpsan-provider";
import { VSpan } from "@/types";

type VSpanProps = {
    plotId: string;
    plotReady: boolean;
    forceUpdate: number;
}

/**
 * Handles the rendering of VSpans onto a specific plot
 * 
 * @param plotId Used to identify the plot that the tooling should be rendered on
 * @param plotReady Signal from main plot that tooling can be drawn
 */
export const VSpans = ({plotId, plotReady} : VSpanProps) => {
    const dragOffset = useRef(0)

    // Hook to trigger the context provider to render context menu
    const {show: showVSpanMenu} = useContextMenu({
        id: VSPAN_MENU_ID
    })

    // Hook to pull in data from context provider
    const {vspans, handleVSpanUpdate, triggerUpdate} = useVSpanContext()

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
            console.error("Could not locate plot to generate vspans")
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
                handleVSpanUpdate()
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
            const range = axis._tmax - axis._tmin
            const upperLimit = axis.d2p(axis._tmax + 2*range)
            const lowerLimit =  axis.d2p(axis._tmin - 2*range)
            const height = lowerLimit - upperLimit

            const xaxis = plot._fullLayout.xaxis;

            const graphGroup = d3.select(overplot)
            graphGroup.selectAll(".vspan").remove() // All VSpans are removed each render cycle

            const drag = d3.drag<SVGRectElement, VSpan>()
                .on("start", function (event, d) {
                    dragOffset.current = xaxis.d2p(d.x) - event.x
                })
                .on("drag", function (event, d) {
                    const newX = event.x + dragOffset.current;
                    d3.select(this).attr("x", newX);

                    const x = xaxis.p2d(newX); // The context provider stores the decimal value rather than pixel
                    d.x = x;
                    handleVSpanUpdate() // Global refresh must be triggered to update all linked plots
                })

            function handleContextMenu(event, vspan: VSpan) {
                showVSpanMenu({
                    event,
                    props: {
                        vspan
                    }
                })
            }

            // Create a line and a transparent drag handle for each VSpan
            for (const vspan of vspans) {
                const x = xaxis.d2p(vspan.x);
                graphGroup.append("line")
                    .attr("class", "vspan")
                    .attr("x1", x)
                    .attr("x2", x)
                    .attr("y1", upperLimit)
                    .attr("y2", upperLimit + height)
                    .attr("stroke", vspan.category.color)
                    .attr("stroke-width", 6)
                    .attr("style", "pointer-events: all")
                    .style("cursor", "move")

                graphGroup.append("rect")
                    .attr("class", "vspan")
                    .attr("x", x-10)
                    .attr("y", upperLimit)
                    .attr("width", 20)
                    .attr("height", height)
                    .attr("fill", "transparent")
                    .attr("style", "pointer-events: all")
                    .style("cursor", "move")
                    .datum(vspan)
                    .call(drag)
                    .on("contextmenu", handleContextMenu)
            }
        }))
    }, [handleVSpanUpdate, plotId, plotReady, showVSpanMenu, vspans, triggerUpdate])

    return (
        <div />
    )
}