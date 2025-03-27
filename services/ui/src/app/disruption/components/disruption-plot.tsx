"use client"

import { ZoneCategory, ZoneShape } from "@/app/components/core/zone";
import { ZoningPlot } from "@/app/components/plots/zoning-plot";
import { ZoneProvider } from "@/app/components/providers/zoning";

type DisruptionInfo = {

    data: Array<{
        time: number,
        value: number
    }>,

    shot_id: string
}


export const DisruptionPlot = ({ data, shot_id }: DisruptionInfo) => {

    const time: number[] = data.map(({ time }) => time);
    const value: number[] = data.map(({ value }) => value);

    const plotData: Plotly.Data[] = [{
        x: time,
        y: value
    }];

    const plotLayout: Partial<Plotly.Layout> = {
        title: {
            text: `Plasma current (${shot_id})`
        },
        xaxis: {
            title: {
                text: 'Time [s]'
            },
        },
        yaxis: {
            title: {
                text: 'ip [A]'
            },
        },
        dragmode: 'pan',
    };

    const plotConfig: Partial<Plotly.Config> = {
        displaylogo: false,
        displayModeBar: true,
        scrollZoom: true,
        modeBarButtonsToRemove: ['toImage', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d'],
    }

    const zoneCategories: ZoneCategory[] = [
        { name: "RampUp", shape: ZoneShape.Rect, color: 'rgb(233, 170, 98)' },
        { name: "FlatTop", shape: ZoneShape.Rect, color: 'rgb(120, 167, 85)' },
        { name: "Disruption", shape: ZoneShape.Line, color: 'rgb(210, 105, 105)' }
    ]

    return (
        <div className="w-100 h-100">
            <ZoneProvider categories={zoneCategories}>
                <ZoningPlot data={plotData} layout={plotLayout} config={plotConfig} />
            </ZoneProvider>
        </div>
    )
};

