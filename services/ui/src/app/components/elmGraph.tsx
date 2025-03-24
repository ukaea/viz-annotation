'use client'
import {useRouter} from "next/navigation"
import { useEffect, useRef, useState, useCallback } from "react"
import { Menu, Item, Submenu, useContextMenu, ItemParams } from 'react-contexify'
import 'react-contexify/ReactContexify.css';
import Plotly from "plotly.js-dist";

var firstDraw = true;
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
    elm_type: string,
    data: Array<{
        time: number,
        value: number,
        ip: number,
    }>,
    shot_id: string
}

export const ElmGraph = ({model_elms, elms, elm_type, data: payload, shot_id} : GraphProps) => {
    const router = useRouter();
    const plotRef = useRef(() => {
        Plotly.newPlot(plotRef.current, [dataTrace, elmTrace, modelElmTrace, ipTrace], layout, config);
        plotRef.current.on('plotly_selected', lassoSelectPeaks);
    });

    const x = payload.map(item => item.time);
    const y = payload.map(item => item.value)

    const elmX = elms.map(item => item.time);
    const elmY = elms.map(item => item.height);
    const elmColors = elms.map(item => item.valid ? 'green' : 'red'); 

    const modelElmX = model_elms.map(item => item.time);
    const modelElmY = model_elms.map(item => item.height);

    var [xElmData, setElmXData] = useState(elmX);
    var [yElmData, setElmYData] = useState(elmY);
    var [colorElmData, setElmColorData] = useState(elmColors);
    var [elmType, setElmType] = useState(elm_type);

    var dataTrace = {
        name: 'Dalpha',
        x: x,
        y: y,
        mode: 'lines',
    };

    var elmTrace = {
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

    var modelElmTrace = {
        name: 'Model ELMs',
        x: modelElmX,
        y: modelElmY,
        marker: {
            size: 9,
            symbol: "diamond-open",
            color: 'purple',
        },
        mode: 'markers',
        type: 'scatter',
        clickmode: 'select'
    };

    var ipTrace = {
        name: 'Ip',
        x: payload.map(item => item.time),
        y: payload.map(item => item.ip),
        xaxis: "x2",
        yaxis: "y2",
        mode: 'lines',
    };

    var layout = {
        uirevision: 'true',
        grid: {rows: 2, columns: 1, pattern: 'independent'},
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

        let points = eventData.points.filter(p => p.curveNumber == 1);
        let selectedIndices = points.map(p => p.pointIndex);
        let colors = [...colorElmData]; 

        console.log(selectedIndices);

        selectedIndices.forEach(index => {
            colors[index] = 'red'; // Change selected points to red
        });

        setElmColorData(colors);
    };

    if (plotRef.current) {
    }

    useEffect(() => {
        if (!plotRef.current) return;
        layout.uirevision = 'true';
        Plotly.react(plotRef.current, [dataTrace, elmTrace, modelElmTrace, ipTrace], layout, config);
        plotRef.current.on('plotly_selected', lassoSelectPeaks);

        return () => {
            if (plotRef.current) {
              plotRef.current.removeAllListeners(); // Clean up event listeners
            }
          };

    }, [xElmData, yElmData, colorElmData]);
    
    // Handles the downloading of the zoning data
    const downloadData = async () => {
        const url = `${process.env.NEXT_PUBLIC_API_URL}/backend-api/annotations/`;

        try {
          const response = await fetch(url);
          if (!response.ok) throw new Error('Network response was not ok');
    
          const data = await response.json();
    
          // Convert JSON to blob and trigger download
          const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
          const link = document.createElement('a');
          link.href = URL.createObjectURL(blob);
          link.download = 'annotations.json';
          link.click();
    
          // Optional: Cleanup
          URL.revokeObjectURL(link.href);
        } catch (error) {
          console.error('Error downloading JSON:', error);
        }
    }


    const saveData = async () => {

        const elmData = xElmData.map((x, index) => ({
            time: xElmData[index],
            height: yElmData[index],
            valid: colorElmData[index] == 'red' ? false : true
        }));

        payload = {
            'shot_id': shot_id,
            'validated': true,
            'elms': elmData,
            'elm_type': elmType,
            'regions': [],
        }

        payload = JSON.stringify(payload);

        const url = `${process.env.NEXT_PUBLIC_API_URL}/backend-api/annotations/${shot_id}`;
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

    // Handle user selecting a radio button
    const handleELMTypeChange = (e) => {
        setElmType(e.target.value);
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
                    <h1 className="text-3xl font-bold text-center text-gray-900">
                        MAST Shot #{shot_id}
                    </h1>
                </header>

                <div class='grid grid-cols-3 space-x-2'>

                    <div class="grid grid-cols-1 toolbar">
                        <fieldset>
                            <legend class='text-center font-bold'>ELM Type:</legend>

                            <div class='space-x-2'>
                                <input type="radio" id="type-none" name="elm-type" value="None"  checked={elmType === 'None'} onChange={handleELMTypeChange}/>
                                <label for="type-none">No ELMs</label>
                            </div>

                            <div class='space-x-2'>
                                <input type="radio" id="type-1" name="elm-type" value="Type I" checked={elmType === 'Type I'} onChange={handleELMTypeChange}/>
                                <label for="type-1">Type I</label>
                            </div>

                            <div class='space-x-2'>
                                <input type="radio" id="type-2" name="elm-type" value="Type II" checked={elmType === 'Type II'} onChange={handleELMTypeChange}/>
                                <label for="type-2">Type II</label>
                            </div>
                            <div class='space-x-2'>
                                <input type="radio" id="type-3" name="elm-type" value="Type III" checked={elmType === 'Type III'} onChange={handleELMTypeChange} />
                                <label for="type-3">Type III</label>
                            </div>
                            <div class='space-x-2'>
                                <input type="radio" id="type-mixed" name="elm-type" value="Mixed" checked={elmType === 'Mixed'} onChange={handleELMTypeChange} />
                                <label for="type-mixed">Mixed</label>
                            </div>
                        </fieldset>
                    </div>


                    <div class="grid grid-cols-1 toolbar">
                            <span class='text-center font-bold'>Peak Params</span>
                            <label for="prominence">Prominence:</label>
                            <input type="range" id="prominence" min="0.001" max="1.0" step="0.1" defaultValue={.5} onMouseUp={handleChangePeakParams}/>

                            <label for="distance">Distance:</label>
                            <input type="range" id="distance" min="1" max="500" step="10" defaultValue={100} onMouseUp={handleChangePeakParams}/>

                    </div>

                    <div class='grid grid-cols-1 space-y-2 toolbar'>
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


                </div>

                <div>
                    <div ref={plotRef} class="w-full">
                    </div>
                </div>

            </div>
        </div>
    )
}