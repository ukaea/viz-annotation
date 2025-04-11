"use client"

import { Category, SpectrogramData } from "@/types"
import { useEffect, useRef, useState } from "react"
import { VSpans } from "@/app/components/tools/vspans"
import { useVSpanContext } from "@/app/components/providers/vpsan-provider"

type LockedModePlotProps = {
    plotId?: string;
    data: SpectrogramData;
    lockedModeCategory: Category;
}

/**
 * Component that handles the plotly and context menu rendering
 * 
 * @param data Saddle coil FFT spectrogram data
 * @param plotId Set plot id externally in case multiple plots are used
 * @param lockedModeCategory Category relating to locked mode event
 */
export const LockedModePlot = ({ data, plotId: externalId, lockedModeCategory }: LockedModePlotProps) => {
    const [updateTools, setUpdateTools] = useState(0)
    const [plotReady, setPlotReady] = useState(false)

    const plotId = externalId || "locked-mode" // Facilitate an external or default ID
    const time = useRef(data.map(({ time }) => time));
    const freq = useRef(data.map(({ frequency }) => frequency));
    const ampl = useRef(data.map(({ amplitude }) => amplitude));

    const { addVSpan } = useVSpanContext()

    const triggerToolUpdate = () => {
        setUpdateTools((current) => (current + 1) % 100)
    }

    // Main plotly rendering
    useEffect(() => {
        const root = document.getElementById(plotId)

        if (!root) {
            console.error("Cannot locate locked-mode element")
            return
        }

        const plotData: Plotly.Data[] = [{
            x: time.current,
            y: freq.current,
            z: ampl.current,
            colorscale: 'YlGnBu',
            type: 'heatmap',
            name: "Saddle Coil FFT"
        }];

        const plotLayout: Partial<Plotly.Layout> = {
            xaxis: {
                title: {
                    text: 'Time [s]'
                },
            },
            yaxis: {
                title: {
                    text: 'Frequency [Hz]'
                },
            },
            showlegend: true,
            dragmode: 'zoom',
        };

        const plotConfig: Partial<Plotly.Config> = {
            displaylogo: false,
            displayModeBar: true,
            scrollZoom: true,
            modeBarButtonsToRemove: ['toImage', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d'],
        }

        const initGraph = async () => {
            const { react } = await import('plotly.js') // Annoyingly there seems to be an issue with plotly so dynamic import is needed

            react(root, plotData, plotLayout, plotConfig).then((plot: Plotly.PlotlyHTMLElement) => {
                const subplot = plot.querySelector(".overplot")?.querySelector(".xy") as HTMLElement
                if (!subplot) {
                    console.error("Cannot locate locked-mode plotly subplot")
                    return
                }

                if (document.getElementsByClassName(`${plotId}-overplot`).length === 0) {
                    const svg = document.createElementNS("http://www.w3.org/2000/svg", "g")
                    svg.setAttribute("class", `${plotId}-overplot`)
                    svg.setAttribute("fill", "none");
                    subplot.appendChild(svg)
                }

                setPlotReady(true)

                plot.on("plotly_relayout", () => {
                    triggerToolUpdate()
                })
            })
        }
        initGraph()

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

        const dragElement = plot.querySelector(".drag")

        if (!dragElement) {
            console.error("Could not locate drag element to assign context menu")
            return
        }

    }, [plotId, plotReady])

    return (
        <div className="w-full px-6 py-3 space-y-3 flex-col">
            {/* Div where plot is inserted */}
            <div id={plotId} className="" />
            <VSpans plotId={plotId} plotReady={plotReady} forceUpdate={updateTools} />
        </div>
    )
}