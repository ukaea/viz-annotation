import { useEffect, useState } from "react"

import * as Plotly from "plotly.js"

import { useZoneContext } from "../providers/zoning"

type PlotInfo = {
    data: Plotly.Data[];
    layout?: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
};

export const ZoningPlot = (info: PlotInfo) => {
    const zoneCtx = useZoneContext()

    useEffect(() => {
        const root = document.getElementById("plot")!;

        Plotly.newPlot(root, info.data, info.layout, info.config).then((plot: Plotly.PlotlyHTMLElement) => {
            zoneCtx.init(plot as HTMLElement)
            zoneCtx.createZones()
            zoneCtx.updateZones(plot)
            zoneCtx.registerEventHandlers(plot)
            plot.on("plotly_relayout", () => {
                zoneCtx.updateZones(plot)
            })
        })


    }, [info]);

    return (
        <div>
            <div id="plot" className="w-100 h-100" />
        </div>
    )
}
