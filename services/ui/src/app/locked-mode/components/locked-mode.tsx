import { LockedModePlot } from "./locked-mode-plot"
import { SpectrogramData, Category, VSpan } from "@/types"
import { VSpanProvider } from "@/app/components/providers/vpsan-provider"

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

    return (
        <div className="flex flex-col items-center space-y-3">
            <header className="p-6">
                <h1 className="text-4xl font-bold text-center text-gray-900">
                    Locked Mode demo
                </h1>
            </header>
            <VSpanProvider categories={lockedModeCategories} initialData={initialLockedMode}>
                <LockedModePlot data={data} lockedModeCategory={lockedModeCategories[0]} />
            </VSpanProvider>
        </div>
    )
}