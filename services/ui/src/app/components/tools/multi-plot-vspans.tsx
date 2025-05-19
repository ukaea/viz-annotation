/* eslint-disable @typescript-eslint/no-explicit-any */
'use client'

import { useEffect, useRef } from 'react'
import { useVSpanContext }   from '@/app/components/providers/vpsan-provider'

type Props = { plotId: string; plotReady: boolean }

export const MultiPlotVSpans = ({ plotId, plotReady }: Props) => {
  /* ------------------------------------------------------------------ */
  const { vspans, triggerUpdate, handleVSpanUpdate } = useVSpanContext() // shared marker data
  const bulkRedraw = useRef(false)                                       // mute flag for relayout loop

  /* helper: every subplot’s x-axis id -------------------------------- */
  const listAxes = (layout: any) =>
    Object.keys(layout)
      .filter(k => /^xaxis(\d*)$/.test(k))
      .map(k => (k === 'xaxis' ? '' : k.replace('xaxis', '')))
      .map(s => (s ? `x${s}` : 'x'))

  /* provider → shape[] ------------------------------------------------ */
  const buildShapes = (layout: any) => {
    const axes = listAxes(layout)
    return vspans.flatMap((sp, idx) =>
      axes.map(xId => ({
        type : 'line',
        uid  : `vspan-${idx}`,                                           // stable id
        xref : xId,                                                      // subplot x-axis
        yref : 'paper',
        x0   : sp.x,                                                     // vertical line at x
        x1   : sp.x,
        y0   : 0,
        y1   : 1,
        line : { width: 2, color: sp.category.color },
        layer: 'above'
      }))
    )
  }

  /* full redraw ------------------------------------------------------- */
  useEffect(() => {
    if (!plotReady) return
    const root = document.getElementById(plotId) as any
    if (!root?._fullLayout) return
    const Plotly: any = require('plotly.js')

    const shapes = buildShapes(root._fullLayout)                         // full replacement (no merge)

    bulkRedraw.current = true
    Plotly.relayout(root, { shapes }).then(() => (bulkRedraw.current = false))
  }, [plotId, plotReady, vspans, triggerUpdate])

  /* live write-back --------------------------------------------------- */
  useEffect(() => {
    if (!plotReady) return
    const root = document.getElementById(plotId) as any
    if (!root?._fullLayout) return

    const onDrag = (ev: any) => {
      if (bulkRedraw.current) return                                    // skip our own relayout
      const key = Object.keys(ev)[0]                                    // first patched prop
      const m   = key?.match(/^shapes\[(\d+)]\.x0$/)                    // we only need x0 (x1 always identical)
      if (!m) return

      const shape = root._fullLayout.shapes?.[+m[1]]
      if (!shape?.uid?.startsWith('vspan-')) return                     // ignore others’ shapes

      const idx  = +shape.uid.split('-')[1]                             // logical marker index
      const newX = ev[key] as number                                    // new data-space x
      if (vspans[idx]) {
        vspans[idx].x = newX                                            // mutate provider
        handleVSpanUpdate()                                             // broadcast change
      }
    }

    root.on('plotly_relayouting', onDrag)                               // listen to live drags
    return () => root.removeListener?.('plotly_relayouting', onDrag)
  }, [plotId, plotReady, vspans, handleVSpanUpdate])

  return null
}
