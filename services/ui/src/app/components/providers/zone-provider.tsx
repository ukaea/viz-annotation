"use client"

import { Zone, Category } from "@/types";
import { createContext, useContext, useEffect, useRef, useState } from "react";
import { Item, ItemParams, Menu, Submenu } from "react-contexify";
import 'react-contexify/ReactContexify.css'
import { useContextMenuProvider } from "./context-menu-provider";

interface ZoneContextInfo {
    zones: Zone[];
    handleZoneUpdate: () => void;
    addZone: (x0: number, x1: number, category: Category) => void;
    triggerUpdate: number;
}

const ZoneContext = createContext<ZoneContextInfo | null>(null)

export const useZoneContext = () => {
    const context = useContext(ZoneContext)
        if (!context) {
            throw new Error("useZoneContext must be used within a ZoneProvider")
        }
        return context
}

export const ZONE_MENU_ID = "zone-provider"

/**
 * Context provider that gives child components shared read/write to zone data
 * 
 * @param categories Array of categories that the zones provided by this context can be
 * @param initialData Array of zones that should be added when initialised
 */
export const ZoneProvider = ({categories, initialData, children} : {
    categories: Category[],
    initialData?: Zone[],
    children: React.ReactNode
}) => {
    const zones = useRef<Zone[]>([])
    const [triggerUpdate, setTriggerUpdate] = useState(0) // Value should be changed to trigger refresh

    const {registerMenuItem} = useContextMenuProvider()
    
    // It is necessary for the context to trigger child refreshes
    const triggerZoneUpdate = () => {
        setTriggerUpdate((current) => (current+1)%10)
    }

    // Provides a method for child components to trigger context refresh
    const handleZoneUpdate = () => {
        triggerZoneUpdate()
    }

    const addZone = (x0: number, x1: number, category: Category) => {
        zones.current.push(
            {
                category,
                x0,
                x1
            }
        )
        triggerZoneUpdate()
    }

    const handleDelete = (input: unknown) => {
        zones.current = zones.current.filter(zone => zone !== input)
        triggerZoneUpdate()
    }

    const handleTypeSetting = ({props}: ItemParams, targetCategory: Category) => {
        zones.current = zones.current.map((zone) => {
            if (zone === props.zone) {
                zone.category = targetCategory
            }
            return zone
        })
        triggerZoneUpdate()
    }

    // On initialisation the tool registers a menu item with the general context menu
    useEffect(() => {
        const add = (x0: number, x1: number, category: Category) => {
            zones.current.push(
                {
                    category,
                    x0,
                    x1
                }
            )
            triggerZoneUpdate()
        }
    
            const addZoneItems = categories.map((category, index) => {
                return (
                    <Item key={`add${index}`} id={`add${index}`} onClick={({props}) => {
                        add(props.x0, props.x1, category)
                    }}>
                        {category.name}
                    </Item>
                )
            })
    
            registerMenuItem("zone", (
                <Submenu key="zone-submenu" label="Add zone">
                    {addZoneItems}
                </Submenu>
            ))
        }, [categories, registerMenuItem])

    // Initialisation of data - this should only run once
    // Effect: run ONCE per mount to populate from initialData
    // – overwrites instead of pushing; cleans on unmount
    useEffect(() => {
        if (!initialData) return;
    
        zones.current = [...initialData]; 
        triggerZoneUpdate();
    
        /* remove stale copy when Strict-Mode unmounts the first render */
        return () => {
          zones.current = [];
        };
      }, [initialData]);

    // Provides an array of the categories for the context menu
    const updateTypeItems = categories.map((category, index) => {
        return (
            <Item key={`update${index}`} id={`update${index}`} onClick={(props) => {handleTypeSetting(props, category)}}>
                {category.name}
            </Item>
        )
    })

    // The context provider is responsible for rendering the context menu relating to zones
    return(
        <ZoneContext.Provider value={{zones: zones.current, handleZoneUpdate, addZone, triggerUpdate}}>
            {children}
            <Menu id={`${ZONE_MENU_ID}`}>
                <Item id="delete" onClick={({props}: ItemParams) => {
                    handleDelete(props.zone)
                }}>
                    Delete
                </Item>
                <Submenu label="Set type">
                    {updateTypeItems}
                </Submenu>
            </Menu>
        </ZoneContext.Provider>
    )

}

