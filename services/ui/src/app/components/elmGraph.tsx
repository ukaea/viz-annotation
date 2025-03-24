'use client'
import {useRouter} from "next/navigation"
import { useEffect, useRef, useState, useCallback } from "react"
import { Menu, Item, Submenu, useContextMenu, ItemParams } from 'react-contexify'
import 'react-contexify/ReactContexify.css';
import Plotly from "plotly.js-dist";
import * as d3 from "d3"

type GraphProps = {
    model_elms: Array<{
        time: number,
        height: number,
        valid: boolean
    }>,
    elms: Array<{
        time: number,
        height: number,
        valid: boolean
    }>,
    data: Array<{
        time: number,
        value: number,
        ip: number,
    }>,
    shot_id: string
}

enum ZoneType {
    Type1,
    Type3
}

type ElmZone = {
    x0: number,
    x1: number,
    start: number,
    type: ZoneType
}

const MENU_ID = "zone_context"

export const ElmGraph = ({model_elms, elms, data: payload, shot_id} : GraphProps) => {
    const router = useRouter();

    const ipPlotRef = useRef(null);
    let isSyncing = false;

    // SVG ref needed by D3 to update graph
    const plotRef = useRef(null);

    // Setup x and y data
    const x = payload.map(item => item.time);
    const y = payload.map(item => item.value)

    const elmX = elms.map(item => item.time);
    const elmY = elms.map(item => item.height);
    const elmColors = elms.map(item => item.valid ? 'green' : 'red'); 

    const modelElmX = model_elms.map(item => item.time);
    const modelElmY = model_elms.map(item => item.height);

    const [xElmData, setElmXData] = useState(elmX);
    const [yElmData, setElmYData] = useState(elmY);
    const [colorElmData, setElmColorData] = useState(elmColors);

    const [xModelElmData, setModelElmXData] = useState(modelElmX);
    const [yModelElmData, setModelElmYData] = useState(modelElmY);

    const [shapes, setShapes] = useState([]);

    useEffect(() => {
        if (plotRef.current) {
            
            const dataTrace = {
                name: 'Dalpha',
                x: x,
                y: y,
                mode: 'lines',
            };

            const elmTrace = {
                name: 'ELMs',
                x: xElmData,
                y: yElmData,
                marker: {
                    size: 5,
                    color: colorElmData,
                },
                mode: 'markers',
                type: 'scatter'
            };

            const modelElmTrace = {
                name: 'Model ELMs',
                x: xModelElmData,
                y: yModelElmData,
                marker: {
                    size: 9,
                    symbol: "diamond-open",
                    color: 'purple',
                },
                mode: 'markers',
                type: 'scatter',
                clickmode: 'select'
            };

            const ipTrace = {
                name: 'Ip',
                x: payload.map(item => item.time),
                y: payload.map(item => item.ip),
                xaxis: "x2",
                yaxis: "y2",
                mode: 'lines',
            };

            const layout = {
                grid: {rows: 2, columns: 1, pattern: 'independent'},
                shapes: shapes,
                dragmode: false,  // Disable default drag behavior
                width: 1500,
                xaxis: {
                    title: {
                      text: 'Time [s]',
                      font: {
                        family: 'Courier New, monospace',
                        size: 12,
                        color: '#7f7f7f'
                      }
                    },
                  },
                yaxis: {
                    title: {
                        text: 'Dalpha [V]',
                        font: {
                        family: 'Courier New, monospace',
                        size: 12,
                        color: '#7f7f7f'
                        }
                    },
                },
                xaxis2: {
                    matches:'x',
                    title: {
                      text: 'Time [s]',
                      font: {
                        family: 'Courier New, monospace',
                        size: 12,
                        color: '#7f7f7f'
                      }
                    },
                  },
                yaxis2: {
                    title: {
                        text: 'Ip [kA]',
                        font: {
                        family: 'Courier New, monospace',
                        size: 12,
                        color: '#7f7f7f'
                        }
                    },
                },
            };

            const clearSelectionIcon = {
                'width': 500,
                'height': 600,
                'path': "M290.7 57.4L57.4 290.7c-25 25-25 65.5 0 90.5l80 80c12 12 28.3 18.7 45.3 18.7L288 480l9.4 0L512 480c17.7 0 32-14.3 32-32s-14.3-32-32-32l-124.1 0L518.6 285.3c25-25 25-65.5 0-90.5L381.3 57.4c-25-25-65.5-25-90.5 0zM297.4 416l-9.4 0-105.4 0-80-80L227.3 211.3 364.7 348.7 297.4 416z"
            };

            const config = {
                displayModeBar: true,
                modeBarButtonsToAdd: [
                  {
                    name: 'Clear Peak Selection',
                    icon: clearSelectionIcon,
                    direction: 'up',
                    click: onClearSelection,
                  },
                ],
                modeBarButtonsToRemove: []}


            // Handle lasso selection of peaks
            function lassoSelectPeaks(eventData) {
                if (!eventData) return; // Ignore if no selection

                console.log(eventData);
                let points = eventData.points.filter(p => p.curveNumber == 1);
                let selectedIndices = points.map(p => p.pointIndex);
                let colors = [...colorElmData]; 

                selectedIndices.forEach(index => {
                    colors[index] = 'red'; // Change selected points to red
                });

                setElmColorData(colors);
            };


            Plotly.newPlot(plotRef.current, [dataTrace, elmTrace, modelElmTrace, ipTrace], layout, config);
            plotRef.current.on('plotly_selected', lassoSelectPeaks);

        }
      }, [xElmData, yElmData, colorElmData, xModelElmData, yModelElmData, shapes]);



    // Handles the downloading of the zoning data
    const downloadData = () => {
        if (spans.current.length === 0) return;

        const csvContent = [
            "x0, x1, type",
            ...spans.current.map(zone => `${zone.x0},${zone.x1},${zone.type}`)
        ].join("\n");

        const blob = new Blob([csvContent], {type: "text/csv"});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a")

        a.href = url
        a.download = `zone_data_${shot_id}.csv`;
        document.body.appendChild(a)
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    const saveData = async () => {

        const elmData = xElmData.map((x, index) => ({
            time: xElmData[index],
            height: yElmData[index],
            valid: colorElmData[index] == 'red' ? false : true
        }));
        console.log(elmData);
        
        payload = {
            'shot_id': shot_id,
            'validated': true,
            'elms': elmData,
            'regions': spans.current.map(zone => ({'time_min': zone.x0, 'time_max': zone.x1, 'type': zone.type}))
        }
        payload = JSON.stringify(payload);

        const url = `${process.env.NEXT_PUBLIC_API_URL}/backend-api/annotations`;
        const response = await fetch(url, {
            method: "POST",
            headers: {
            "Content-Type": "application/json",
            },
            body: payload,
        });
    }

    const nextShot = async () => {
      saveData();
      const next_shot_id = await queryNextShot();
      router.push(`/${Number(next_shot_id)}`)
    }

    const queryNextShot = async () => {
        const url = `${process.env.NEXT_PUBLIC_API_URL}/backend-api/next`;
        const response = await fetch(url, {method: "GET"});
        const data = await response.json();
        return data['shot_id'];
    }

    const handleChangePeakParams = async (event) => {
        onClearSelection(event);

        const prominence = parseFloat(document.getElementById('prominence').value);
        const distance = parseFloat(document.getElementById('distance').value);

        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/backend-api/annotations/${shot_id}?method=classic&prominence=${prominence}&distance=${distance}&force=${true}`)
        const annotationData = await response.json();

        const elmX = annotationData.elms.map(item => item.time);
        const elmY = annotationData.elms.map(item => item.height);

        setElmXData(elmX);
        setElmYData(elmY);
    };

    // Handle clearing selection of peaks
    function onClearSelection(eventData) {
        const colors = elmColors.fill('green'); 
        setElmColorData(colors);
    };

    return (
        <div style={{ display: "flex" }}>
            <div class="flex flex-col items-center space-y-3">

                <header className="p-6">
                    <h1 className="text-4xl font-bold text-center text-gray-900">
                        ELM Tagging
                    </h1>
                </header>

                <div class='toolbar space-x-2'>
                    <button class='btn-primary'
                        onClick={downloadData}
                    >Download Labels</button>

                    <button class="btn-primary"
                        onClick={saveData}
                    >Save Labels</button>

                    <button class="btn-primary"
                        onClick={nextShot}
                    >Next Shot</button>
                </div>


                <div ref={plotRef} class="w-full">
                </div>
                <div ref={ipPlotRef} class="w-full">
                </div>


                <div class='grid grid-cols-2 space-x-2'>
                    <div class="grid grid-cols-1 toolbar">
                            <span class='text-center font-bold'>Peak Params</span>
                            <label for="prominence">Prominence:</label>
                            <input type="range" id="prominence" min="0.001" max="1.0" step="0.1" defaultValue={.5} onMouseUp={handleChangePeakParams}/>

                            <label for="distance">Distance:</label>
                            <input type="range" id="distance" min="1" max="500" step="10" defaultValue={100} onMouseUp={handleChangePeakParams}/>

                    </div>

                    <div class="grid grid-cols-1 toolbar">
                        <fieldset>
                            <legend class='text-center font-bold'>ELM Types:</legend>

                            <div class='space-x-2'>
                                <input type="radio" id="type-none" name="elm-type" />
                                <label for="type-none">No ELMs</label>
                            </div>

                            <div class='space-x-2'>
                                <input type="radio" id="type-1" name="elm-type" />
                                <label for="type-1">Type I</label>
                            </div>

                            <div class='space-x-2'>
                                <input type="radio" id="type-2" name="elm-type" />
                                <label for="type-2">Type II</label>
                            </div>
                            <div class='space-x-2'>
                                <input type="radio" id="type-3" name="elm-type" />
                                <label for="type-3">Type III</label>
                            </div>
                            <div class='space-x-2'>
                                <input type="radio" id="type-mixed" name="elm-type"  />
                                <label for="type-mixed">Mixed</label>
                            </div>
                        </fieldset>
                    </div>

                </div>

            </div>


            <Menu id={MENU_ID}>
                {/* <Item id="delete" onClick={handleDelete}>Delete</Item> */}
                <Submenu label="Set type">
                    <Item id="zone1" onClick={({props}: ItemParams) => {
                        handleTypeSetting(props.zone, ZoneType.Type1)
                    }}>Zone I</Item>
                    <Item id="zone3" onClick={({props}: ItemParams) => {
                        handleTypeSetting(props.zone, ZoneType.Type3)
                    }}>Zone III</Item>
                </Submenu>
            </Menu>
        </div>
    )
}