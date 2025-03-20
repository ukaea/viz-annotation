import { useEffect } from "react";
import * as d3 from "d3"
import { useGraph } from "../tooling/zoning-graph";

export const Line = ({ data, id }: { data: { time: number; value: number }[], id: string }) => {
    const { graphGroup, xScale, yScale } = useGraph();

    useEffect(() => {
        if (!graphGroup) return;

        const line = d3.line<{time: number; value: number}>()
            .x(d => xScale(d.time))
            .y(d => yScale(d.value));

        const g = d3.select(graphGroup)
        g.selectAll(`.${id}`).remove()
        g.append("path")
            .attr("class", id)
            .datum(data)
            .attr("fill", "none")
            .attr("stroke", "blue")
            .attr("stroke-width", 2)
            .attr("d", line);
    }, [data, graphGroup, id, xScale, yScale]);

    return null;
};