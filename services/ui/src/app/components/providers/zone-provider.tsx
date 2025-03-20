import React, { createContext, useContext, useRef, useState } from "react"
import { Item, ItemParams, Menu, Submenu } from "react-contexify"

/**
 * Possible zone types that can be assigned in the annotator
 */
export enum ZoneType {
    Type1,
    Type3
}

/**
 * Structure containing information needed to define a zone
 */
export type ElmZone = {
    x0: number,
    x1: number,
    start: number,
    type: ZoneType
}

/**
 * Props to be included in the ZoneContext
*/
interface ZoneContextProps {
    zones: ElmZone[]
    handleZoneUpdate: (update: ElmZone[] | null) => void
    updateZones: boolean
    handleDelete: (input: unknown) => void
    handleTypeSetting: (targetZone: ElmZone, targetType: ZoneType) => void
}
const ZoneContext = createContext<ZoneContextProps | null>(null)

export const ELM_MENU_ID = "elm_zone_context"

/**
 * Gives children components access to a zone context, which allows multiple graphs to share zoning data
 */
export const ZoneProvider = ({ children }: { children: React.ReactNode }) => {
    const zones = useRef<ElmZone[]>([]) // Reference to the zone list
    const [updateZones, setUpdateZones] = useState(false) // Toggled to trigger zone refresh

    /**
     * Used to trigger a zone update on all components using the zone provider
     * @param update Can be set to an ELMZone list to force an update directly or null to just trigger an update
     */
    const handleZoneUpdate = (update: ElmZone[] | null) => {
        if (update) {
            zones.current = update
        }
        setUpdateZones((current) => !current)
    }

    const handleDelete = (input: unknown) => {
        zones.current = zones.current.filter(zone => zone !== input)
        setUpdateZones((current) => !current)
    }

    const handleTypeSetting = (targetZone: ElmZone, targetType: ZoneType) => {
        zones.current = zones.current.map((zone) => {
            if (zone === targetZone) {
                zone.type = targetType
            }
            return zone
        })
        setUpdateZones((current) => !current)
    }

    const handleContextDelete = ({props}: ItemParams) => {
        if (props.zone) {
            handleDelete(props.zone)
        }
    };

    return (
        <ZoneContext.Provider value={{zones: zones.current, handleZoneUpdate, updateZones, handleDelete, handleTypeSetting}}>
            {children}
            <Menu id={ELM_MENU_ID}>
                <Item id="delete" onClick={handleContextDelete}>Delete</Item>
                <Submenu label="Set type">
                    <Item id="zone1" onClick={({props}: ItemParams) => {
                        handleTypeSetting(props.zone, ZoneType.Type1)
                    }}>Zone I</Item>
                    <Item id="zone3" onClick={({props}: ItemParams) => {
                        handleTypeSetting(props.zone, ZoneType.Type3)
                    }}>Zone III</Item>
                </Submenu>
            </Menu>
        </ZoneContext.Provider>
    )
}

/**
 * Hook to the context provided by ZoneProvider - enforces use inside the correct provider component
 * @returns ZoneContext which can be destructured inside child components
 */
export const useZones = () => {
    const context = useContext(ZoneContext);
    if (!context) {
        throw new Error("useZone must be used within a ZoneProvider")
    }
    return context
}