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
        const overplot = document.getElementsByClassName(`${plotId}-overplot`)[0]

        // Rendering should not be attempted if the required handles are not found
        if (!plot) {
            console.error("Could not locate plot to generate vspans")
            return
        }
        if (!overplot) {
            console.error("Could not locate D3 overplot to generate vspans")
            return
        }

        // Finds the axis data which is used for scaling drawings - if subplots are used additional logic is needed
        const marginTop = plot._fullLayout._size.t
        const totalHeight = plot._fullLayout.yaxis._length
        const y = marginTop - totalHeight
        const height = 2 * totalHeight

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
                .attr("y1", y)
                .attr("y2", y + height)
                .attr("stroke", vspan.category.color)
                .attr("stroke-width", 6)
                .attr("style", "pointer-events: all")
                .style("cursor", "move")

            graphGroup.append("rect")
                .attr("class", "vspan")
                .attr("x", x-10)
                .attr("y", y)
                .attr("width", 20)
                .attr("height", height)
                .attr("fill", "transparent")
                .attr("style", "pointer-events: all")
                .style("cursor", "move")
                .datum(vspan)
                .call(drag)
                .on("contextmenu", handleContextMenu)
        }
    }, [handleVSpanUpdate, plotId, plotReady, showVSpanMenu, vspans, triggerUpdate])

    return (
        <div />
    )
}