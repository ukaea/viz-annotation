"use client"

import React, { createContext, useCallback, useContext, useState } from "react"
import { Menu, ShowContextMenuParams, useContextMenu } from "react-contexify";

type MakeOptional<Type, Key extends keyof Type> = Omit<Type, Key> & Partial<Pick<Type, Key>>;

interface ContextMenuContextType {
    registerMenuItem: (id: string, element: React.ReactNode) => void;
    show: (params: MakeOptional<ShowContextMenuParams<unknown>, "id">) => void
}

const ContextMenuContext = createContext<ContextMenuContextType | null>(null);

export const useContextMenuProvider = () => {
    const context = useContext(ContextMenuContext);
    if (!context) {
        throw new Error("useRegisterContextMenuItem must be used within a ContextMenuProvider")
    }
    return context
}

/**
 * Context provider that gives child components access to context menu data
 * 
 * @param menuId Allows for a unique id to be assigned to the context menu 
 */
export const ContextMenuProvider = ({menuId, children} : {
    menuId: string, 
    children: React.ReactNode
}) => {
    const [menuElements, setMenuElements] = useState<Map<string, React.ReactNode>>(new Map());
    const {show} = useContextMenu({ id:  menuId})

    // Allows tools to register their own menu item in the general context menu
    const registerMenuItem = useCallback((id: string, element: React.ReactNode) => {
        setMenuElements((prev) => {
            if (prev.has(id)) return prev;
            const newMap = new Map(prev)
            newMap.set(id, element)
            return newMap;
        })
    }, [])

    return (
        <ContextMenuContext.Provider value={{registerMenuItem, show}}>
            {children}
            <Menu id={menuId}>
                {[...menuElements.values()]}
            </Menu>
        </ContextMenuContext.Provider>
    )
}