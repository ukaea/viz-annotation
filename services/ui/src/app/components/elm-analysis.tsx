'use client'
import * as d3 from "d3"
import { ZoomProvider } from "./providers/zoom-provider"
import { useZones, ZoneProvider } from "./providers/zone-provider"
import { ZoningGraph } from "./tooling/zoning-graph"
import { Line } from "./base-graphs/line-graph"
import { VSpan } from "./base-graphs/vspan-graph"
import {useRouter} from "next/navigation"

type GraphProps = {
    elms: Array<{
        time: number,
        height: number,
        valid: boolean
    }>,
    data: Array<{
        time: number,
        value: number 
    }>,
    frequency: Array<{
        time: number,
        value: number
    }>
    shot_id: string
}

/**
 * Provides an example of how a group of graphs could be set up making use of the linked zoom and zoning
 */
export const ElmAnalysis = ({elms, data: payload, frequency, shot_id} : GraphProps) => {

    const width = 2000, height = 400, margin = 50

    const time_extent = d3.extent(payload, d => d.time) as [number, number]
    const data_extent = d3.extent(payload, d => d.value) as [number, number]

    const elmSpans = elms.map(elm => (
        {
            time: elm.time
        }
    ))

    return (
        <div>
            <ZoomProvider width={width}  margin={margin} domain={time_extent}>
                <ZoneProvider>
                    <div className="flex flex-col items-center space-y-3">
                        <header className="p-6">
                            <h1 className="text-4xl font-bold text-center text-gray-900">
                                ELM Tagging Demo
                            </h1>
                        </header>
                        <ZoningGraph height={height} yDomain={data_extent}>
                            <Line data={payload} id={"dalpha"}/>
                            <VSpan data={elmSpans} id={"elm_spans"}/>
                        </ZoningGraph>
                        <ZoningGraph height={height} yDomain={d3.extent(frequency, d => d.value) as [number, number]}>
                            <Line data={frequency} id={"elm_frequency"}/>
                        </ZoningGraph>
                        <ElmToolbar elms={elms} shot_id={shot_id}/>
                    </div>
                </ZoneProvider>
            </ZoomProvider>
        </div>
    )
}

/**
 * Toolbar used for navigating MAST shots - this must be a seperate component in order to use zone context
 */
const ElmToolbar = ({shot_id, elms} : {
    shot_id: string, 
    elms: Array<{
        time: number,
        height: number,
        valid: boolean
    }>
}) => {
    const router = useRouter();

    const {zones} = useZones()

    // Handles the downloading of the zoning data
    const downloadData = () => {
        if (zones.length === 0) return;

        const csvContent = [
            "x0, x1, type",
            ...zones.map(zone => `${zone.x0},${zone.x1},${zone.type}`)
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
        const payload = {
            'shot_id': shot_id,
            'elms': elms,
            'regions': zones.map(zone => ({'time_min': zone.x0, 'time_max': zone.x1, 'type': zone.type}))
        }

        const url = `${process.env.NEXT_PUBLIC_API_URL}/db-api/shots`;
        await fetch(url, {
            method: "POST",
            headers: {
            "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });
    }

    const nextShot = async () => {
      router.push(`/${Number(shot_id)+1}`)
    }

    const previousShot = async () => {
      router.push(`/${Number(shot_id)-1}`)
    }

    return (
        <div className='toolbar'>
            <button className="btn-primary"
                onClick={previousShot}
            >Previous Shot</button>

            <button className='btn-primary'
                onClick={downloadData}
            >Download Labels</button>

            <button className="btn-primary"
                onClick={saveData}
            >Save Labels</button>

            <button className="btn-primary"
                onClick={nextShot}
            >Next Shot</button>
        </div>
    )
}