import { useEffect, useState } from "react"

import * as Plotly from "plotly.js"

import { useZoneContext } from "../providers/zoning"
import { ZoneTable } from "../core/zone";

type PlotInfo = {
    data: Plotly.Data[];
    layout?: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
};

export const ZoningPlot = (info: PlotInfo) => {
    const zoneCtx = useZoneContext()

    const [shouldUpdate, setShouldUpdate] = useState(true)

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
            setShouldUpdate(false)
        })
    }, [info, zoneCtx]);

    return (
        <div className="w-full px-6 py-3 space-y-3 flex-col">
            <div id="plot" className="" />
            {!shouldUpdate && <ZoneTable zones={zoneCtx.zones.current} />}
        </div>
    )
}
