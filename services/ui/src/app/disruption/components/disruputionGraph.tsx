'use client'
import { useState, useEffect } from 'react';
import * as Plotly from "plotly.js";

type GraphProps = {

    data: Array<{
        time: number,
        value: number
    }>,

    shot_id: string
}

enum ZoneType {
    RampUp,
    FlatTop
}

type Zone = {
    x0: number,
    x1: number,
    type: ZoneType
}

export const DisruptionGraph = ({ data, shot_id }: GraphProps) => {

    const time: number[] = data.map(({ time }) => time);
    const value: number[] = data.map(({ value }) => value);
    const rectMargin = 1.05;

    const tMin = Math.min(...time);
    const tMax = Math.max(...time);
    const vMin = Math.min(...value);
    const vMax = Math.max(...value);

    const [rampUp, setRampUp] = useState<Zone>({ x0: 0.2 * tMax, x1: 0.3 * tMax, type: ZoneType.RampUp });
    const [flatTop, setFlatTop] = useState<Zone>({ x0: 0.5 * tMax, x1: 0.6 * tMax, type: ZoneType.FlatTop });
    const [disruptPoint, setDisruptPoint] = useState<number>(0.75 * tMax);

    useEffect(() => {
        interface PlotHTMLElement extends HTMLElement {
            on(eventName: string, handler: Function): void;
        }
        const plotElement = document.getElementById('plot')! as PlotHTMLElement;

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

        Plotly.newPlot(plotElement, plotData, plotLayout, plotConfig);

        plotElement.on('plotly_click', () => {
            alert("You clicked the plot!");
        });

        const update: Partial<Plotly.Layout> = {};
        update.shapes = [];
        update.shapes.push({
            opacity: 0.5,
            type: 'rect',
            x0: rampUp.x0,
            x1: rampUp.x1,
            y0: vMin * rectMargin,
            y1: vMax * rectMargin,
            fillcolor: 'rgb(76, 171, 235)',
            line: {
                width: 0,
            }
        })
        update.shapes.push({
            opacity: 0.5,
            type: 'rect',
            x0: flatTop.x0,
            x1: flatTop.x1,
            y0: vMin * rectMargin,
            y1: vMax * rectMargin,
            fillcolor: 'rgb(121, 236, 173)',
            line: {
                width: 0,
            }
        })
        update.shapes.push({
            opacity: 0.5,
            type: 'line',
            x0: disruptPoint,
            x1: disruptPoint,
            y0: vMin * rectMargin,
            y1: vMax * rectMargin,
            line: {
                color: 'rgb(211, 29, 22)',
                width: 2,
            }
        })

        Plotly.relayout(plotElement, update);
    });

    return (
        <div id="plot"></div>
    )
};

