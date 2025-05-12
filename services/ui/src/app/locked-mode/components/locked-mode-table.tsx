"use client"

import { useVSpanContext } from "@/app/components/providers/vpsan-provider"
import { useZoneContext } from "@/app/components/providers/zone-provider"
import { Category } from "@/types"
import { JSX, useEffect, useState } from "react"

export const LockedModeTable = () => {
    const [entries, setEntries] = useState<JSX.Element[]>([])

    const { zones, triggerUpdate: triggerZoneUpdate } = useZoneContext()
    const { vspans: lockedModes, triggerUpdate: triggerLockedModeUpdate } = useVSpanContext()

    useEffect(() => {
        const entriesBuffer: JSX.Element[] = []

        const generateEntry = (id: string, x0: string, x1: string, marker: string, category: Category) => (
            <tr key={id} className="bg-white border-b dark:bg-gray-800 dark:border-gray-700 border-gray-200 hover:bg-gray-100 dark:hover:bg-gray-600">
                <th scope="row" className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap dark:text-white">
                    <div className="flex space-x-3 items-center" style={{ cursor: "pointer" }}>
                        <div className={marker} style={{ background: category.color }} />
                        <span>{category.name}</span>
                    </div>
                </th>
                <td className="px-6 py-4">
                    <span>{x0}</span>
                </td>
                <td className="px-6 py-4">
                    <span>{x1}</span>
                </td>
            </tr >
        )

        for (const [index, zone] of zones.entries()) {
            const x0 = zone.x0.toFixed(6)
            const x1 = zone.x1.toFixed(6)
            const marker = "w-5 h-5 sm:rounded-lg"

            entriesBuffer.push(generateEntry(`zone-${index}`, x0, x1, marker, zone.category))
        }

        for (const [index, lockedMode] of lockedModes.entries()) {
            const x0 = lockedMode.x.toFixed(6)
            const x1 = "--"
            const marker = "w-5 h-1.5 sm:rounded-lg"

            entriesBuffer.push(generateEntry(`lockedMode-${index}`, x0, x1, marker, lockedMode.category))
        }

        setEntries(entriesBuffer)
    }, [setEntries, zones, triggerZoneUpdate, lockedModes, triggerLockedModeUpdate])

    return (
        <div className="relative w-fit overflow-x-auto shadow-md sm:rounded-lg ml-auto mr-auto">
            <table className="w-full text-sm text-left rtl:text-right text-gray-500 dark:text-gray-400">
                <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
                    <tr>
                        <th scope="col" className="px-6 py-3">
                            Zone category
                        </th>
                        <th scope="col" className="px-6 py-3">
                            x0
                        </th>
                        <th scope="col" className="px-6 py-3">
                            x1
                        </th>
                    </tr>
                </thead>
                <tbody>
                    {entries}
                </tbody>
            </table>
        </div>
    )
}