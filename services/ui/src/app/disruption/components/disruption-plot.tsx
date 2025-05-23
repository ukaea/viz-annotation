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

                if (!subplot.querySelector(`.${plotId}-overplot-xy`)) { // ensure only one custom overlay group is present
                    const svg = document.createElementNS("http://www.w3.org/2000/svg", "g")
                    svg.setAttribute("class", `${plotId}-overplot-xy`) // xy is currently hardcoded as only one subplot
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

        function handleContextMenu(event, plot) {
            const xaxis = plot._fullLayout.xaxis;
            const bb = event.target.getBoundingClientRect();
            const x0 = xaxis.p2d(event.clientX - bb.left);
            const x1 = xaxis.p2d(event.clientX - bb.left + 100);

            showContextMenuRef.current({
                event,
                props: {
                    x0,
                    x1
                }
            })
        }

        const dragElement = plot.querySelector(".drag")

        if (!dragElement) {
            console.error("Could not locate drag element to assign context menu")
            return
        }

        const contextHandler = (event) => { //  wrap handler so we can remove it
            handleContextMenu(event, plot)
        } 

        dragElement.addEventListener("contextmenu", contextHandler) // add context-menu listener

        return () => { // remove listener on effect cleanup
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