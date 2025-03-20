'use client'
import { use, useEffect } from 'react';
import * as Plotly from "plotly.js";

type GraphProps = {

    data: Array<{
        time: number,
        value: number
    }>,

    shot_id: string
}

export const DisruptionGraph = ({ data, shot_id }: GraphProps) => {

    useEffect(() => {
        interface PlotHTMLElement extends HTMLElement {
            on(eventName: string, handler: Function): void;
        }
        const plotElement = document.getElementById('testPlot')! as PlotHTMLElement;

        const plotData = [{
            x: data.map(({ time }) => time),
            y: data.map(({ value }) => value)
        }];

        const plotLayout = {
            title: {
                text: `Plasma current (${shot_id})`
            },
            xaxis: {
                title: {
                    text: 'Time [s]'
                }
            },
            yaxis: {
                title: {
                    text: 'ip [A]'
                }
            },
            margin: { t: 100 }
        };

        const plotConfig = {
            displaylogo: false
        };

        Plotly.newPlot(plotElement, plotData, plotLayout, plotConfig);

        plotElement.on('plotly_click', () => {
            alert("You clicked the plot!");
        });
    });

    return (
        <div>
            {/* <h2>Disruption ({shot_id})</h2> */}
            <div id="testPlot"></div>
            {/* <p style={{ whiteSpace: 'pre-line' }}>
                {data.map(({ time, value }) => `time: ${time}, value: ${value}`).join('\n')}
            </p> */}
        </div>
    )
};

