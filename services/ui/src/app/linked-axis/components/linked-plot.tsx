"use client"

import { useContextMenuProvider } from '@/app/components/providers/context-menu-provider'
import { VSpans } from '@/app/components/tools/vspans'
import { Zones } from '@/app/components/tools/zones'
import { useEffect, useRef, useState } from 'react'

export const LinkedPlot = () => {
    const [updateTools, setUpdateTools] = useState(0)
    const [plotReady, setPlotReady] = useState(false)

    const plotId = "linked"

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

        const x_data = [0, 1, 2, 3, 4, 5]

        const trace1: Plotly.Data = {
            x: x_data,
            y: x_data.map(n => n**2),
            line: {
                color: "black"
            },
            name: "ip"
        };

        const trace2: Plotly.Data = {
            x: x_data,
            y: x_data.map(n => 2*n),
            line: {
                color: "black"
            },
            name: "dalpha",
            xaxis: 'x',
            yaxis: 'y2'
        };

        const plotData = [trace1, trace2]
    
        const plotLayout: Partial<Plotly.Layout> = {
            grid: {
                rows: 2,
                columns: 1,
                subplots: ['xy', 'xy2']
            }
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
                plotElement = plot; // save reference to remove listeners later

                ["xy", "xy2"].forEach(coordinateSystem => {
                    const subplot = plot.querySelector(`.subplot.${coordinateSystem}`)?.querySelector(".overplot")?.querySelector(`.${coordinateSystem}`) as HTMLElement
                    if (!subplot) {
                        console.error("Cannot locate disruption plotly subplot")
                        return
                    }

                    if (!subplot.querySelector(`.${plotId}-overplot-${coordinateSystem}`)) { // ensure only one custom overlay group is present
                        const svg = document.createElementNS("http://www.w3.org/2000/svg", "g")
                        svg.setAttribute("class", `${plotId}-overplot-${coordinateSystem}`)
                        svg.setAttribute("fill", "none");
                        subplot.appendChild(svg)
                    }
                });

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
            <Zones plotId={plotId} subplotId='xy2' plotReady={plotReady} forceUpdate={updateTools} />
        </div>
    )
}