"use client"

import { Category, VSpan } from "@/types"
import React, { createContext, useContext, useEffect, useRef, useState } from "react"
import { Item, ItemParams, Menu, Submenu } from "react-contexify"

interface VSpanContextInfo {
    vspans: VSpan[];
    handleVSpanUpdate: () => void;
    addVSpan: (x: number, category: Category) => void;
    triggerUpdate: number;
}

const VSpanContext = createContext<VSpanContextInfo | null>(null)

export const useVSpanContext = () => {
    const context = useContext(VSpanContext)
    if (!context) {
        throw new Error("useVSpanContext must be used within a VSpanProvider")
    }
    return context
}

export const VSPAN_MENU_ID = "vspan-provider"

/**
 * Context provider that gives child components shared read/write to vspan data
 * 
 * @param categories Array of categories that the vspans provided by this context can be
 * @param initialData Array of vspans that should be added when initialised
 */
export const VSpanProvider = ({categories, initialData, children} : {
    categories: Category[],
    initialData?: VSpan[],
    children: React.ReactNode
}) => {
    const spans = useRef<VSpan[]>([])
    const [triggerUpdate, setTriggerUpdate] = useState(0) // Value should be changed to trigger refresh

    // It is necessary for the context to trigger child refreshes
    const triggerVSpanUpdate = () => {
        setTriggerUpdate((current) => (current+1)%10)
    }

    // Provides a method for child components to trigger context refresh
    const handleVSpanUpdate = () => {
        triggerVSpanUpdate()
    }

    const addVSpan = (x: number, category: Category) => {
        spans.current.push({
            category,
            x
        })
        triggerVSpanUpdate()
    }

    const handleDelete = (input: unknown) => {
        spans.current = spans.current.filter(span => span !== input)
        triggerVSpanUpdate()
    }

    const handleTypeSetting = ({props}: ItemParams, targetCategory: Category) => {
        spans.current = spans.current.map((span) => {
            if (span === props.vspan) {
                span.category = targetCategory
            }
            return span
        })
        triggerVSpanUpdate()
    }

    // Initialisation of data - this should only run once
    useEffect(() => {
        if (!initialData) return

        for (const span of initialData) {
            spans.current.push(span)
        }
        triggerVSpanUpdate()
    }, [initialData])

    // Provides an array of the categories for the context menu
    const updateTypeItems = categories.map((category, index) => {
        return (
            <Item key={`update${index}`} id={`update${index}`} onClick={(props) => {handleTypeSetting(props, category)}}>
                {category.name}
            </Item>
        )
    })

    // The context provider is responsible for rendering the context menu relating to VSpans
    return (
        <VSpanContext.Provider value={{vspans: spans.current, handleVSpanUpdate, addVSpan, triggerUpdate}}>
            {children}
            <Menu id={`${VSPAN_MENU_ID}`}>
                <Item id="delete" onClick={({props}: ItemParams) => {
                    handleDelete(props.vspan)
                }}>
                    Delete
                </Item>
                {updateTypeItems.length > 1 && (
                    <Submenu label="Set type">
                        {updateTypeItems}
                    </Submenu>
                )}
                
            </Menu>
        </VSpanContext.Provider>
    )
}