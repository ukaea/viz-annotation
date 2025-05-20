import { ContextMenuProvider } from "@/app/components/providers/context-menu-provider"
import { ZoneProvider } from "@/app/components/providers/zone-provider"
import { LinkedPlot } from "./linked-plot"
import { Category } from "@/types"

export const LinkedDemo = () => {
    const zoneCategories: Category[] = [
                { name: "RampUp", color: 'rgb(233, 170, 98)' },
                { name: "FlatTop", color: 'rgb(120, 167, 85)' },
                { name: "RampDown", color: 'rgb(108, 189, 224)' }
            ]
            
    return (
        <div className="flex flex-col items-center space-y-3">
            <header className="p-6">
                <h1 className="text-4xl font-bold text-center text-gray-900">
                    Shared Axis Demo
                </h1>
            </header>
            <ContextMenuProvider menuId="disruption-menu">
                <ZoneProvider categories={zoneCategories}>
                    <LinkedPlot />
                </ZoneProvider>
            </ContextMenuProvider>
        </div>
    )
}