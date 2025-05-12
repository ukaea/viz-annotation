import { LockedModePlot } from "./locked-mode-plot"
import { LockedModeTable } from "./locked-mode-table"
import { SpectrogramData, Category, VSpan, Zone } from "@/types"
import { VSpanProvider } from "@/app/components/providers/vpsan-provider"
import { ContextMenuProvider } from "@/app/components/providers/context-menu-provider"
import { ZoneProvider } from "@/app/components/providers/zone-provider"

type LockedModeInfo = {
    data: SpectrogramData
}

export const LockedMode = ({ data }: LockedModeInfo) => {

    const lockedModeCategories: Category[] = [
        { name: "Locked Mode", color: "rgb(255, 0, 0)" },
    ]

    const initialLockedMode: VSpan[] = [
        { x: 0.1, category: lockedModeCategories[0] },
    ]

    const zoneCategories: Category[] = [
        { name: "ZoneA", color: 'rgb(255, 0, 0)' },
    ]

    const initialZones: Zone[] = [
        { x0: 0.4, x1: 0.5, category: zoneCategories[0] },
    ]

    return (
        <div className="flex flex-col items-center space-y-3">
            <header className="p-6">
                <h1 className="text-4xl font-bold text-center text-gray-900">
                    Locked Mode demo
                </h1>
            </header>
            <ContextMenuProvider menuId="locked-mode-menu">
                <VSpanProvider categories={lockedModeCategories} initialData={initialLockedMode}>
                    <ZoneProvider categories={zoneCategories} initialData={initialZones}>
                        <LockedModePlot data={data} lockedModeCategory={lockedModeCategories[0]} />
                        <LockedModeTable />
                    </ZoneProvider>
                </VSpanProvider>
            </ContextMenuProvider>
        </div >
    )
}