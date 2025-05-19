/* eslint-disable @typescript-eslint/no-explicit-any */
// Synchronizes semi-transparent rectangular “zones” across all subplots of a Plotly figure.
// Rebuilds the full set of shapes on provider updates and mirrors drag/resize events back to the ZoneProvider.
'use client'

import { useEffect, useRef } from 'react'
import { useZoneContext }    from '@/app/components/providers/zone-provider'

type Props = { plotId: string; plotReady: boolean }                     // prop that identifies the Plotly div + a flag from parent

export const MultiPlotZones = ({ plotId, plotReady }: Props) => {
  /* ------------------------------------------------------------------ */
  const { zones, triggerUpdate, handleZoneUpdate } = useZoneContext()   // access shared zone data & notifier
  const bulkRedraw = useRef(false)                                      // true while *our* relayout is running (prevents echo)

  /* ---------- helper: list every subplot’s axis ids & domains ------- */
  const listSubplots = (layout: any) =>                                 // layout == Plotly._fullLayout
    Object.keys(layout)
      .filter(k => /^xaxis(\d*)$/.test(k))                              // keep xaxis / xaxis2 / …
      .map(k => {
        const s = k === 'xaxis' ? '' : k.replace('xaxis', '')           // suffix '' | '2' | '3' …
        const x = s ? `x${s}` : 'x'                                     // x , x2 , x3 …
        const y = s ? `yaxis${s}` : 'yaxis'                             // yaxis , yaxis2 , …
        return { xId: x, yDomain: layout[y]?.domain as [number, number] }
      })

  /* ---------- helper: convert provider → Plotly shape objects ------- */
  const buildShapes = (layout: any) => {
    const axes = listSubplots(layout)                                   // [{xId:'x',yDomain:[0,0.5]}, …]
    return zones.flatMap((z, idx) =>                                    // one logical zone → N rectangles (one per subplot)
      axes.map(({ xId, yDomain }) => ({
        type : 'rect',                                                  // Plotly shape kind
        uid  : `zone-${idx}`,                                           // stable id to find siblings later
        xref : xId,                                                     // attach to that subplot’s x-axis
        yref : 'paper',                                                 // full axis height
        x0   : z.x0,                                                    // data-space coords
        x1   : z.x1,
        y0   : yDomain[0],                                              // domain → [0,0.5] etc.
        y1   : yDomain[1],
        line : { width: 0 },                                            // no outline
        fillcolor: z.category.color,                                    // colour from provider
        opacity  : 0.25,
        layer: 'below'                                                  // under the traces
      }))
    )
  }

  /* ---------- effect #1 – draw / refresh --------------------------- */
  useEffect(() => {
    if (!plotReady) return                                              // wait until Plotly is initialised
    const root = document.getElementById(plotId) as any
    if (!root?._fullLayout) return
    const Plotly: any = require('plotly.js')

    const shapes = buildShapes(root._fullLayout)                        // rebuild *all* of our rectangles

    bulkRedraw.current = true                                           // mute relayouting handler
    Plotly.relayout(root, { shapes }).then(() => (bulkRedraw.current = false))
  }, [plotId, plotReady, zones, triggerUpdate])                         // runs whenever provider bumps triggerUpdate

  /* ---------- effect #2 – write-back while user drags -------------- */
  useEffect(() => {
    if (!plotReady) return
    const root = document.getElementById(plotId) as any
    if (!root?._fullLayout) return

    const onRelayout = (ev: any) => {                                   // ev = diff object from Plotly
      if (bulkRedraw.current) return                                    // ignore our own relayout diff

      const pending: Record<number, { x0?: number; x1?: number }> = {}  // accumulate x0/x1 patches per zoneIdx

      Object.entries(ev).forEach(([k, v]) => {
        const m = k.match(/^shapes\[(\d+)]\.(x0|x1)$/)                  // matches 'shapes[3].x0'
        if (!m) return
        const shape = root._fullLayout.shapes?.[+m[1]]                  // find shape referenced by index
        if (!shape?.uid?.startsWith('zone-')) return                    // skip other tools’ shapes
        const idx = +shape.uid.split('-')[1]                            // logical zone index
        ;(pending[idx] ??= {})[m[2] as 'x0' | 'x1'] = v as number        // stash new x0/x1
      })

      let dirty = false
      Object.entries(pending).forEach(([i, u]) => {                     // apply merged updates to provider
        const z = zones[+i]
        if (!z) return
        if (u.x0 !== undefined) z.x0 = u.x0
        if (u.x1 !== undefined) z.x1 = u.x1
        dirty = true
      })
      if (dirty) handleZoneUpdate()                                     // notify provider listeners
    }

    root.on('plotly_relayouting', onRelayout)                           // continuous while dragging
    return () => root.removeListener?.('plotly_relayouting', onRelayout)
  }, [plotId, plotReady, zones, handleZoneUpdate])

  return null                                                           // overlay is managed entirely by Plotly shapes
}
