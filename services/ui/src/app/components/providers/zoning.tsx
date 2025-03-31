import React, { useContext, createContext, useRef } from "react";

import { Zone, ZoneCategory } from "../core/zone";

interface ZoneContextInfo {
    zones: React.RefObject<Zone[]>
    categories: ZoneCategory[]
    group: React.RefObject<SVGGElement | null>
}

const ZoneContext = createContext<ZoneContextInfo | null>(null)

export const useZoneContext = () => {
    const context = useContext(ZoneContext)
    if (!context) {
        throw new Error("useZone must be used within a ZoneProvider")
    }
    return context
}

export const ZoneProvider = ({ categories, children }: { categories: ZoneCategory[], children: React.ReactNode }) => {
    const zones = useRef<Zone[]>([])
    const group = useRef<SVGGElement | null>(null)

    return (
        <ZoneContext.Provider value={{
            zones,
            categories,
            group: group,
        }}>
            {children}
        </ZoneContext.Provider>
    )
}
