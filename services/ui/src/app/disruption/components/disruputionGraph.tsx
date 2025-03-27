'use client'
import { useState, useEffect, useRef, use } from 'react';
import * as Plotly from "plotly.js";
import * as d3 from 'd3';

type GraphProps = {

    data: Array<{
        time: number,
        value: number
    }>,

    shot_id: string
}

enum ZoneType {
    Rect,
    Line
}

type Zone = {
    type: ZoneType,
    x0: number,
    x1?: number,
    isDragging: boolean,
    isResizingLeft?: boolean,
    isResizingRight?: boolean,
    dragElement: SVGRectElement | SVGLineElement | null,
    resizeElements?: SVGLineElement[] | null,
}

const zoneDragHandler = (zone: Zone, clientX: number) => {
    zone.isDragging = true;
    const newOffsetX = clientX - parseFloat(zone.dragElement!.getAttribute('x')!);
    return {
        zone: zone,
        offsetX: newOffsetX
    }
}

function guidGenerator() {
    var S4 = function () {
        return (((1 + Math.random()) * 0x10000) | 0).toString(16).substring(1);
    };
    return (S4() + S4() + "-" + S4() + "-" + S4() + "-" + S4() + "-" + S4() + S4() + S4());
}

interface PlotHTMLElement extends HTMLElement {
    on(eventName: string, handler: Function): void;
}

type PlotInfo = {
    id: string,
    root: Plotly.Root,
    data: Plotly.Data[],
    layout?: Partial<Plotly.Layout>,
    config?: Partial<Plotly.Config>
}

const plotWithZones = (info: PlotInfo, zones: Zone[]) => {
    const plotElement = document.getElementById(info.id)! as PlotHTMLElement;

    Plotly.newPlot(plotElement, info.data, info.layout, info.config).then((plot: any) => {
        const overplot = plotElement.querySelector('.overplot')! as HTMLElement;
        const overplot_xy = overplot.querySelector('.xy')! as HTMLElement;

        var d3group: SVGGElement;
        if (document.getElementsByClassName('d3zone_overplot').length > 0) {
            d3group = document.getElementsByClassName('d3zone_overplot')[0] as SVGGElement;
        } else {
            d3group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            d3group.setAttribute('class', 'd3zone_overplot');
            d3group.setAttribute('fill', 'none');
            d3group.setAttribute('stroke', 'red');
            d3group.setAttribute('stroke-width', '2');
            overplot_xy.appendChild(d3group);
        }

        const svg = d3.select(d3group);
        svg.selectAll("*").remove();

        const setupZone = (zone: Zone) => {
            const guid = guidGenerator();
            const zoneRect = svg.append("rect").attr("class", "zone-" + guid);
            const zoneRectElement = zoneRect.node() as SVGRectElement;
            zoneRectElement.style.pointerEvents = 'all';

            zone.dragElement = zoneRectElement;

            if (zone.type === ZoneType.Rect) {
                const zoneLeft = svg.append("line").attr("class", "zone-" + guid + "-l");
                const zoneLeftElement = zoneLeft.node() as SVGLineElement;
                zoneLeftElement.style.pointerEvents = 'all';

                const zoneRight = svg.append("line").attr("class", "zone-" + guid + "-r");
                const zoneRightElement = zoneRight.node() as SVGLineElement;
                zoneRightElement.style.pointerEvents = 'all';

                zone.resizeElements = [zoneLeftElement, zoneRightElement];
            }

            return zone;
        }
        for (var zone of zones) {
            zone = setupZone(zone);
        }

        const updateZone = (zone: Zone) => {

            return zone
        }
    });
}


export const DisruptionGraph = ({ data, shot_id }: GraphProps) => {

    const time: number[] = data.map(({ time }) => time);
    const value: number[] = data.map(({ value }) => value);

    const [rampUp, setRampUp] = useState<Zone>({ type: ZoneType.Rect, x0: 0, x1: 0.05, isDragging: false, isResizingLeft: false, isResizingRight: false, dragElement: null, resizeElements: null });
    const [flatTop, setFlatTop] = useState<Zone>({ type: ZoneType.Rect, x0: 0.1, x1: 0.15, isDragging: false, isResizingLeft: false, isResizingRight: false, dragElement: null, resizeElements: null });
    const [disruption, setDisruption] = useState<Zone>({ type: ZoneType.Line, x0: 0.2, isDragging: false, dragElement: null });

    const [mouseOffsetX, setMouseOffsetX] = useState<number>(0);

    useEffect(() => {
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

        Plotly.newPlot(plotElement, plotData, plotLayout, plotConfig).then(
            (plot: any) => {
                const overplot = plotElement.querySelector('.overplot')! as HTMLElement;
                const overplot_xy = overplot.querySelector('.xy')! as HTMLElement;

                var d3group: SVGGElement;
                if (document.getElementsByClassName('d3zone_overplot').length > 0) {
                    d3group = document.getElementsByClassName('d3zone_overplot')[0] as SVGGElement;
                } else {
                    d3group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                    d3group.setAttribute('class', 'd3zone_overplot');
                    d3group.setAttribute('fill', 'none');
                    d3group.setAttribute('stroke', 'red');
                    d3group.setAttribute('stroke-width', '2');
                    overplot_xy.appendChild(d3group);
                }

                const svg = d3.select(d3group);
                svg.selectAll("*").remove();

                const rampUpRect = svg.append("rect").attr("class", "rampup-zone");
                const rampUpRectL = svg.append("line").attr("class", "rampup-zone-l");
                const rampUpRectR = svg.append("line").attr("class", "rampup-zone-r");
                const rampUpEl = rampUpRect.node() as SVGRectElement;
                const rampUpElL = rampUpRectL.node() as SVGLineElement;
                const rampUpElR = rampUpRectR.node() as SVGLineElement;
                rampUpEl.style.pointerEvents = 'all';
                rampUpElL.style.pointerEvents = 'all';
                rampUpElR.style.pointerEvents = 'all';


                const updateZone = (zone: Zone,
                    rect: d3.Selection<SVGRectElement, unknown, null, undefined>,
                    left: d3.Selection<SVGLineElement, unknown, null, undefined>,
                    right: d3.Selection<SVGLineElement, unknown, null, undefined>) => {
                    const xRange = plot._fullLayout.xaxis.range;
                    const xScale = plot._fullLayout.xaxis._length / (xRange[1] - xRange[0]);

                    const marginTop = plot._fullLayout._size.t;
                    const totalHeight = plot._fullLayout.yaxis._length;

                    const x0px = (zone.x0 - xRange[0]) * xScale;
                    const widthPx = (zone.x1 - zone.x0) * xScale;

                    const y = marginTop - totalHeight;
                    const rectHeight = 2 * totalHeight; // adjust as needed

                    rect
                        .attr("x", x0px)
                        .attr("y", y)
                        .attr("width", widthPx)
                        .attr("height", rectHeight)
                        .attr("fill", "rgb(76, 171, 235)")
                        .attr("opacity", 0.5)
                        .style("cursor", "move");

                    left
                        .attr("x1", x0px)
                        .attr("x2", x0px)
                        .attr("y1", y)
                        .attr("y2", y + rectHeight)
                        .attr("stroke", "black")
                        .attr("stroke-width", 3)
                        .style("cursor", "ew-resize");

                    right
                        .attr("x1", x0px + widthPx)
                        .attr("x2", x0px + widthPx)
                        .attr("y1", y)
                        .attr("y2", y + rectHeight)
                        .attr("stroke", "black")
                        .attr("stroke-width", 3)
                        .style("cursor", "ew-resize");
                }

                updateZone(rampUp, rampUpRect, rampUpRectL, rampUpRectR);

                const mouseDownHandlerDragging = (event: MouseEvent) => {
                    setIsDragging(true);
                    setOffsetX(event.clientX - parseFloat(rampUpEl.getAttribute('x')!));
                }
                const mouseDownHandlerResizingLeft = (event: MouseEvent) => {
                    setIsResizingLeft(true);
                    setOffsetX(event.clientX - parseFloat(rampUpElL.getAttribute('x1')!));
                }
                const mouseDownHandlerResizingRight = (event: MouseEvent) => {
                    setIsResizingRight(true);
                    setOffsetX(event.clientX - parseFloat(rampUpElR.getAttribute('x1')!));
                }
                const mouseMoveHandler = (event: MouseEvent) => {
                    const xScale = plot._fullLayout.xaxis._length / (plot._fullLayout.xaxis.range[1] - plot._fullLayout.xaxis.range[0]);
                    const mouseX = event.clientX - mouseOffsetX;

                    if (isDragging) {
                        const currentWidth = rampUp.x1 - rampUp.x0;
                        const newX0 = plot._fullLayout.xaxis.range[0] + mouseX / xScale;
                        setRampUp({ x0: newX0, x1: newX0 + currentWidth, type: ZoneType.RampUp });
                        updateZone({ x0: newX0, x1: newX0 + currentWidth, type: ZoneType.RampUp }, rampUpRect, rampUpRectL, rampUpRectR);
                    }
                    if (isResizingLeft) {
                        const newX0 = plot._fullLayout.xaxis.range[0] + mouseX / xScale;
                        setRampUp({ x0: newX0, x1: rampUp.x1, type: ZoneType.RampUp });
                        updateZone({ x0: newX0, x1: rampUp.x1, type: ZoneType.RampUp }, rampUpRect, rampUpRectL, rampUpRectR);
                    }
                    if (isResizingRight) {
                        const newX1 = plot._fullLayout.xaxis.range[0] + mouseX / xScale;
                        setRampUp({ x0: rampUp.x0, x1: newX1, type: ZoneType.RampUp });
                        updateZone({ x0: rampUp.x0, x1: newX1, type: ZoneType.RampUp }, rampUpRect, rampUpRectL, rampUpRectR);
                    }

                    document.removeEventListener('mousemove', mouseMoveHandler);
                }
                const mouseUpHandler = () => {
                    setIsDragging(false);
                    setIsResizingLeft(false);
                    setIsResizingRight(false);
                }

                rampUpEl.addEventListener('mousedown', mouseDownHandlerDragging);
                rampUpElL.addEventListener('mousedown', mouseDownHandlerResizingLeft);
                rampUpElR.addEventListener('mousedown', mouseDownHandlerResizingRight);
                document.addEventListener('mousemove', mouseMoveHandler);
                document.addEventListener('mouseup', mouseUpHandler);

                plotElement.on('plotly_relayout', () => {
                    updateZone(rampUp, rampUpRect, rampUpRectL, rampUpRectR);
                });
            }
        );
    });

    return (
        <div>
            <div id="plot" className="h-100"></div>
            <div className="overflow-x-auto bg-white">
                <table className="min-w-full text-left text-sm whitespace-nowrap">

                    <thead className="uppercase tracking-wider border-b-2">
                        <tr>
                            <th scope="col" className="px-6 py-4 w-1/3">
                                Zone
                            </th>
                            <th scope="col" className="px-6 py-4 w-1/3">
                                t0 [s]
                            </th>
                            <th scope="col" className="px-6 py-4 w-1/3">
                                t1 [s]
                            </th>
                        </tr>
                    </thead>

                    <tbody>
                        <tr className="border-b">
                            <th scope="row" className="px-6 py-4">
                                Ramp up
                            </th>
                            <td className="px-6 py-4">{rampUp.x0}</td>
                            <td className="px-6 py-4">{rampUp.x1}</td>
                        </tr>

                        <tr className="border-b">
                            <th scope="row" className="px-6 py-4">
                                Flat top
                            </th>
                            <td className="px-6 py-4">{flatTop.x0}</td>
                            <td className="px-6 py-4">{flatTop.x1}</td>
                        </tr>

                        <tr className="border-b">
                            <th scope="row" className="px-6 py-4">
                                Disruption
                            </th>
                            <td className="px-6 py-4">{disruptPoint}</td>
                            <td className="px-6 py-4">-</td>
                        </tr>

                    </tbody>
                </table>
            </div>
        </div>
    )
};

