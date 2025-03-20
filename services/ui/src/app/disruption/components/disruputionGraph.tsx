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
        const plotElement: HTMLElement = document.getElementById('testPlot')!;

        Plotly.newPlot(plotElement, [{
            x: data.map(({ time }) => time),
            y: data.map(({ value }) => value)
        }], {
            margin: { t: 0 }
        });
    });

    return (
        <div>
            <h2>Disruption ({shot_id})</h2>
            <div id="testPlot"></div>
            <p style={{ whiteSpace: 'pre-line' }}>
                {data.map(({ time, value }) => `time: ${time}, value: ${value}`).join('\n')}
            </p>
        </div>
    )
};

