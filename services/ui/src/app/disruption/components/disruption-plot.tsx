"use client"

import { useContextMenuProvider } from "@/app/components/providers/context-menu-provider"
import { VSpans } from "@/app/components/tools/vspans"
import { Zones } from "@/app/components/tools/zones"
import { Category, TimeSeriesData } from "@/types"
import { useEffect, useRef, useState } from "react"

type DisruptionPlotProps = {
    plotId?: string;
    data: TimeSeriesData;
    zoneCategories: Category[];
    disruptionCategory: Category;
}

/**
 * Component that handles the plotly and context menu rendering
 * 
 * @param data Disruption time series data
 * @param plotId Set plot id externally in case multiple plots are used
 * @param zoneCategories Zone categories to display in context menu
 * @param disruptionCategory Category relating to disruption
 */
export const DisruptionPlot = ({data, plotId: externalId} : DisruptionPlotProps) => {
    const [updateTools, setUpdateTools] = useState(0)
    const [plotReady, setPlotReady] = useState(false)

    const plotId =  externalId || "disruption" // Facilitate an external or default ID
    const time = useRef(data.map(({ time }) => time));
    const value = useRef(data.map(({ value }) => value));

    const {show: showContextMenu} = useContextMenuProvider()
    const showContextMenuRef = useRef(showContextMenu)

    const triggerToolUpdate = () => {
        setUpdateTools((current) => (current + 1) % 100)
    }

    // Main plotly rendering
    useEffect(() => {
        const root = document.getElementById(plotId)

        if (!root) {
            console.error("Cannot locate disruption element")
            return
        }

        const plotData: Plotly.Data[] = [{
            x: time.current,
            y: value.current,
            line: {
                color: "black"
            },
            name: "ip"
        }];
    
        const plotLayout: Partial<Plotly.Layout> = {
            xaxis: {
                title: {
                    text: 'Time [s]'
                },
            },
            yaxis: {
                title: {
                    text: 'Plasma current, ip [A]'
                },
            },
            showlegend: true,
            dragmode: 'pan',
        };
    
        const plotConfig: Partial<Plotly.Config> = {
            displaylogo: false,
            displayModeBar: true,
            scrollZoom: false,
        }

        let plotElement: Plotly.PlotlyHTMLElement | null = null // holds the created plot for later cleanup

        const initGraph = async () => {
            const { react } = await import('plotly.js') // Annoyingly there seems to be an issue with plotly so dynamic import is needed

            react(root, plotData, plotLayout, plotConfig).then((plot: Plotly.PlotlyHTMLElement) => {
                plotElement = plot // save reference to remove listeners later

                const subplot = plot.querySelector(".overplot")?.querySelector(".xy") as HTMLElement
                if (!subplot) {
                    console.error("Cannot locate disruption plotly subplot")
                    return
                }

                if (!subplot.querySelector(`.${plotId}-overplot`)) { // ensure only one custom overlay group is present
                    const svg = document.createElementNS("http://www.w3.org/2000/svg", "g")
                    svg.setAttribute("class", `${plotId}-overplot`)
                    svg.setAttribute("fill", "none");
                    subplot.appendChild(svg)
                }

                setPlotReady(true)

                const relayoutHandler = () => { // triggers re-render of overlay tools when axes change
                    triggerToolUpdate()
                } 
                plot.on("plotly_relayout", relayoutHandler) // attach listener so it can be removed
            })
        }
        initGraph()
        
        return () => { // cleanup on unmount / Fast-Refresh
            plotElement?.removeAllListeners?.("plotly_relayout"); // detach relayout listener
            root?.querySelector(`.${plotId}-overplot`)?.remove(); // remove custom overlay group
            setPlotReady(false); // reset ready state
        } 
    }, [plotId])

    // Handles context menu creation
    useEffect(() => {
        if (!plotReady) {
            // Plot may not have loaded yet - this will rerun after loading
            return
        }

        const plot = document.getElementById(plotId)

        if (!plot) {
            console.error("Could not locate plot to assign context menu")
            return
        }

        /**
         * Builds the props packet for the Context-Menu.
         * We now include BOTH:
         *   – the new generic fields  { x, y, xScale, yScale }
         *   – the legacy helpers      { x0, x1 }  (100-px wide slice)
         * so that pre-existing menu items that still expect x0/x1 keep working.
         */
        function handleContextMenu(event: MouseEvent, plot) {
            const xaxis = plot._fullLayout.xaxis
            const yaxis = plot._fullLayout.yaxis

            const bb    = (event.target as HTMLElement).getBoundingClientRect()
            const relX  = event.clientX - bb.left
            const relY  = event.clientY - bb.top

            const x      = xaxis.p2d(relX)
            const y      = yaxis.p2d(relY)
            const xScale = Math.abs(xaxis.d2p(1) - xaxis.d2p(0))   // px / unit
            const yScale = Math.abs(yaxis.d2p(1) - yaxis.d2p(0))

            /* legacy helpers – 100-pixel-wide default zone */
            const unitWidth = 100 / xScale
            const x0 = x
            const x1 = x + unitWidth

            showContextMenuRef.current({
                event,
                props: { x, y, xScale, yScale, x0, x1 }            // merged packet
            })
        }


        const dragElement = plot.querySelector(".drag")
        if (!dragElement) {
            console.error("Could not locate drag element to assign context menu")
            return
        }

        const contextHandler = (event: MouseEvent) => handleContextMenu(event, plot)
        dragElement.addEventListener("contextmenu", contextHandler)

        return () => {
            dragElement.removeEventListener("contextmenu", contextHandler)
        }
    }, [plotId, plotReady])

    return (
        <div className="w-full px-6 py-3 space-y-3 flex-col">
            {/* Div where plot is inserted */}
            <div id={plotId} className="" />
            <Zones plotId={plotId} plotReady={plotReady} forceUpdate={updateTools} />
            <VSpans plotId={plotId} plotReady={plotReady} forceUpdate={updateTools} />
        </div>
    )
}