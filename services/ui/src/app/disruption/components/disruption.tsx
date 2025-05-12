import { Category, VSpan, Zone } from "@/types"
import { DisruptionPlot } from "./disruption-plot"
import { ZoneProvider } from "@/app/components/providers/zone-provider"
import { VSpanProvider } from "@/app/components/providers/vpsan-provider"
import { DisruptionTable } from "./disruption-table"
import { ContextMenuProvider } from "@/app/components/providers/context-menu-provider"

type DisruptionInfo = {

    data: Array<{
        time: number,
        value: number
    }>
}

/**
 * Handles the creation of the zoning and disruption context providers as well as any necessary plots and tables
 * 
 * @param data Time series data relating to the plasma current 
 * @returns 
 */
export const Disruption = ({ data }: DisruptionInfo) => {
    const zoneCategories: Category[] = [
            { name: "RampUp", color: 'rgb(233, 170, 98)' },
            { name: "FlatTop", color: 'rgb(120, 167, 85)' },
            { name: "RampDown", color: 'rgb(108, 189, 224)' }
        ]

    const initialZones: Zone[] = [
        { x0: 0.05, x1: 0.1, category: zoneCategories[0] },
        { x0: 0.15, x1: 0.2, category: zoneCategories[1] },
    ]

    const disruptionCategories: Category[] = [
            { name: "Disruption", color: 'rgb(255, 0, 0)' },
        ]

    const initialDisruption: VSpan[] = [
        { x: 0.3, category: disruptionCategories[0] }
    ]

    return (
        <div className="flex flex-col items-center space-y-3">
            <header className="p-6">
                <h1 className="text-4xl font-bold text-center text-gray-900">
                    Ramp-up / Flat-top / Disruption point Demo
                </h1>
            </header>
            <ContextMenuProvider menuId="disruption-menu">
                <VSpanProvider categories={disruptionCategories} initialData={initialDisruption}>
                    <ZoneProvider categories={zoneCategories} initialData={initialZones}>
                        <DisruptionPlot data={data} zoneCategories={zoneCategories} disruptionCategory={disruptionCategories[0]}/>
                        <DisruptionTable />
                    </ZoneProvider>
                </VSpanProvider>
            </ContextMenuProvider>
        </div>
    )
}