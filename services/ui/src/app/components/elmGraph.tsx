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
        value: number 
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

    // SVG ref needed by D3 to update graph
    const plotRef = useRef(null)

    const spans = useRef<ElmZone[]>([])

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

    // const handleMouseDown = useCallback(async (event) => {
    //     var rect = plotRef.current.getBoundingClientRect();
    //     var xPixel = event.clientX - rect.left;
    //     var yPixel = event.clientY - rect.top;

    //     // Get the current axis ranges
    //     var xAxis = plotRef.current._fullLayout.xaxis;
    //     var yAxis = plotRef.current._fullLayout.yaxis;

    //     // Convert pixel to data coordinates
    //     var xData = xAxis.p2d(xPixel);
    //     var yData = yAxis.p2d(yPixel);

    //     console.log('Mouse Down at:', { x: xData, y: yData });
    //     var coords = {x: xData, y: yData};
    //     if (!coords) return;
    
    //     const newShape = {
    //       type: "rect",
    //       x0: coords.x - 0.5, // Slightly offset
    //       x1: coords.x + 0.5,
    //       y0: coords.y - 0.5,
    //       y1: coords.y + 0.5,
    //       line: { color: "red", width: 2 },
    //     };

    //       // Update the state to include the new shape
    //       setShapes((prevShapes) => [...prevShapes, newShape]);
    // }, []);


    useEffect(() => {
        if (plotRef.current) {
            
            const dataTrace = {
                name: 'Dalpha',
                x: x,
                y: y,
                mode: 'lines',
                type: 'scatter'
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

            const layout = {
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
                }
                }
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


            Plotly.newPlot(plotRef.current, [dataTrace, elmTrace, modelElmTrace], layout, config);
            plotRef.current.on('plotly_selected', lassoSelectPeaks);
            // plotRef.current.addEventListener("mousedown", handleMouseDown);

        }
      }, [xElmData, yElmData, colorElmData, xModelElmData, yModelElmData, shapes]);


    // // Set up D3 scale bars - refs to allow useEffects to track and update them
    // const width = 1300, height = 400, margin = 50

    // const time_extent = d3.extent(payload, d => d.time) as [number, number]
    // const xScale = useRef(d3.scaleLinear().domain(time_extent).range([margin, width - margin]));
    // const xScaleZoomedRef = useRef(xScale.current.copy()) // Copy required for zooming (investigate this)

    // const value_extent = d3.extent(payload, d => d.value) as [number, number]
    // const yScale = useRef(d3.scaleLinear().domain(value_extent).range([height - margin, margin]));

    // Tracks the zones that have been added - can be monitored by zone useEffect
    const [updateZones, setUpdateZones] = useState(false)

    // // Context menu set up including event callbacks
    // const {show} = useContextMenu({
    //     id: MENU_ID
    // })

    // const handleDelete = ({props}: ItemParams) => {
    //     if (props.zone) {
    //         spans.current = spans.current.filter(span => span !== props.zone);
    //         setUpdateZones(update => !update);
    //     }
    // };

    // const handleTypeSetting = (target_zone: ElmZone, type: ZoneType) => {
    //     spans.current = spans.current.map((zone) => {
    //         if (zone === target_zone) {
    //             zone.type = type
    //         }
    //         return zone
    //     })
    //     setUpdateZones((update) => !update)
    // }

    // Allows handles to resize zones - useEffect dependency handles redrawing
    // const resizeDrag = d3.drag<Element, ElmZone>()
    //     .on("drag", function(event, d) {
    //         const handleType = d3.select(this).attr("data-handle");
    //         const newX = xScale.current.invert(event.x)

    //         if (handleType === "left") {
    //             d.x0 = Math.min(newX, d.x1 - 0.005);  // Prevent overlap (this should be revisted)
    //         } else {
    //             d.x1 = Math.max(newX, d.x0 + 0.005);
    //         }
    //         setUpdateZones((update) => !update)
    //     })

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


    // useEffect(() => {
    //     const svg = d3.select(svgRef.current)
    //         .attr("width", width)
    //         .attr("height", height)

    //     svg.selectAll("*").remove(); // Clear previous contents in case of re-draw

    //     // Clip path prevents rendering of graph outside of axes when scaled / panned
    //     svg.append("clipPath")
    //         .attr("id", "clip")
    //         .append("rect")
    //         .attr("x", margin)
    //         .attr("y", margin)
    //         .attr("width", width - 2 * margin)
    //         .attr("height", height - 2 * margin);

    //     const clipGroup = svg.append("g")
    //         .attr("clip-path", "url(#clip)")
    //     const graphGroup = clipGroup.append("g")
    //         .attr("class", "graph-content")
               
    //     const xAxis = svg.append("g")
    //         .attr("transform", `translate(0, ${height - margin})`)
    //         .call(d3.axisBottom(xScale.current));

    //     const yAxis = svg.append("g")
    //         .attr("transform", `translate(${margin}, 0)`)
    //         .call(d3.axisLeft(yScale.current));

    //     const line = d3.line<{time: number; value: number}>()
    //         .x(d => xScale.current(d.time))
    //         .y(d => yScale.current(d.value));

    //     graphGroup.append("path")
    //         .datum(payload)
    //         .attr("fill", "none")
    //         .attr("stroke", "blue")
    //         .attr("stroke-width", 2)
    //         .attr("d", line);

    //     elms.forEach(element => {
    //         graphGroup.append("line")
    //             .attr("x1", xScale.current(element.time))
    //             .attr("y1", yScale.current(d3.min(payload, d => d.value)))
    //             .attr("x2", xScale.current(element.time))
    //             .attr("y2", yScale.current(d3.max(payload, d => d.value)))
    //             .attr("stroke", "red")
    //             .attr("stroke-width", 1)
    //             .attr("stroke-dasharray", "4,4"); // Optional: dashed line
            
    //     });

    //     // Handles the drawing of new zones
        // const drag = d3.drag()
        //     .on("start", function (event) {
        //         const startX = xScale.current.invert(event.x);
        //         spans.current.push({ x0: startX, x1: startX, start: startX, type: ZoneType.Type1 })
        //         setUpdateZones((update) => !update)
        //     })
        //     .on("drag", function (event) {
        //         const span = spans.current[spans.current.length - 1]
        //         const dragValue = xScale.current.invert(event.x);
        //         if (dragValue < span.start) {
        //             span.x0 = dragValue;
        //             span.x1 = span.start;
        //         } else {
        //             span.x0 = span.start;
        //             span.x1 = dragValue;
        //         }
        //         setUpdateZones((update) => !update)
        //     })
        //     .on("end", function () {
        //         const span = spans.current[spans.current.length - 1]
        //         if (span.x0 > span.x1) [span.x0, span.x1] = [span.x1, span.x0];
        //         if (span.x1 - span.x0 < 0.01) {
        //             spans.current.pop();
        //         }
        //         setUpdateZones((update) => !update)
        //     });

    //     // Handles zoom and panning
    //     const zoom = d3.zoom()
    //         .scaleExtent([1, 5])
    //         .translateExtent([[margin, margin], [width - margin, height - margin]])
    //         .on("zoom", (event) => {
    //             graphGroup.attr("transform", event.transform);
    //             xScaleZoomedRef.current = event.transform.rescaleX(xScale.current);
    //             xAxis.call(d3.axisBottom(xScaleZoomedRef.current));
    //             yAxis.call(d3.axisLeft(event.transform.rescaleY(yScale.current)));
                
    //             const scaleFactor = 1 / event.transform.k
    //             graphGroup.selectAll("path")
    //                 .attr("stroke-width", 2 * scaleFactor)
    //                 setUpdateZones((update) => !update)
    //         });

    //     // Initially set up the graph with the zone creation functionality
    //     let isShiftPressed = false;
            // d3.select(plotRef.current).call(drag) // Typing of these calls should be visted

    //     // If the shift key is pressed change to panning functionality
    //     const keydownHandler = (event: KeyboardEvent) => {
    //         if (event.key === "Shift" && !isShiftPressed) {
    //             isShiftPressed = true;
    //             svg.on(".drag", null);
    //             svg.call(zoom);
    //         }
    //     };

    //     // Return to zoning after shift is released
    //     const keyupHandler = (event: KeyboardEvent) => {
    //         if (event.key === "Shift") {
    //             isShiftPressed = false;
    //             svg.on(".zoom", null);
    //             svg.call(drag);
    //         }
    //     };

    //     document.addEventListener("keydown", keydownHandler);
    //     document.addEventListener("keyup", keyupHandler);

    //     // Ensure listeners are cleaned up correctly
    //     return () => {
    //         document.removeEventListener("keydown", keydownHandler);
    //         document.removeEventListener("keyup", keyupHandler);
    //     };

    // }, [payload])

    // Zone rendering
    // useEffect(() => {
    //     // Required to allow context menu to be triggered
    //     function handleContextMenu(event, zone: ElmZone) {
    //         show({
    //             event,
    //             props: {
    //                 key: 'value',
    //                 zone,
    //             }
    //         })
    //     }

    //     // Pulls the graph group from the SVG to ensure zones are correctly drawn
    //     const graphGroup = d3.select(plotRef.current).select(".graph-content")

    //     // Data binding to efficiently handles creation / update / deletion of zones
    //     const spanGroups = graphGroup.selectAll(".span-group").data(spans.current);

    //     let height = 3;
    //     const newGroups = spanGroups.enter()
    //         .append("g")
    //         .attr("class", "span-group");

    //     newGroups.append("rect")
    //         .attr("class", "span")
    //         .merge(spanGroups.select(".span"))
    //         .attr("x", d => xScale.current(d.x0))
    //         .attr("y", 0)
    //         .attr("width", d => xScale.current(d.x1) - xScale.current(d.x0))
    //         .attr("height", height)
    //         .attr("fill", d => {
    //             if (d.type == ZoneType.Type1) return "rgba(0, 0, 255, 0.3)";
    //             if (d.type == ZoneType.Type3) return "rgba(0, 255, 0, 0.3)";
    //             return "rgba(255, 0, 0, 0.3)";
    //         })
    //         .on("contextmenu", handleContextMenu); // Assigns context menu callback to each zone

    //     newGroups.append("circle")
    //         .attr("class", "handle left-handle")
    //         .attr("data-handle", "left")
    //         .attr("r", 5)
    //         .attr("fill", "red")
    //         .call(resizeDrag)
    
    //     newGroups.append("circle")
    //         .attr("class", "handle right-handle")
    //         .attr("data-handle", "right")
    //         .attr("r", 5)
    //         .attr("fill", "red")
    //         .call(resizeDrag)

    //     spanGroups.select(".span")
    //         .attr("x", d => xScale.current(d.x0))
    //         .attr("width", d => xScale.current(d.x1) - xScale.current(d.x0));
        
    //     spanGroups.select(".left-handle")
    //         .attr("cx", d => xScale.current(d.x0))
    //         .attr("cy", height / 2);
        
    //     spanGroups.select(".right-handle")
    //         .attr("cx", d => xScale.current(d.x1))
    //         .attr("cy", height / 2);

    //     spanGroups.exit().remove();
    // }, [resizeDrag, show, updateZones])

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

                <div class="grid grid-cols-1 toolbar">
                        <span class='text-center font-bold'>Peak Params</span>
                        <label for="prominence">Prominence:</label>
                        <input type="range" id="prominence" min="0.001" max="1.0" step="0.1" defaultValue={.5} onMouseUp={handleChangePeakParams}/>

                        <label for="distance">Distance:</label>
                        <input type="range" id="distance" min="1" max="500" step="10" defaultValue={100} onMouseUp={handleChangePeakParams}/>
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