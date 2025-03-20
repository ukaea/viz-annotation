import { useEffect } from "react";
import * as d3 from "d3"
import { useGraph } from "../tooling/zoning-graph";

export const VSpan = ({ data, id }: { data: { time: number; }[], id: string }) => {
    const { graphGroup, xScale, yScale } = useGraph();

    useEffect(() => {
        if (!graphGroup) return;

        const g = d3.select(graphGroup)
        const boundingBox = g.node()?.getBoundingClientRect()
        const height = boundingBox ? boundingBox?.height : 0
        g.selectAll(`.${id}`).remove()
        data.forEach(element => {
            g.append("line")
                .attr("class", id)
                .attr("x1", xScale(element.time))
                .attr("y1", 0)
                .attr("x2", xScale(element.time))
                .attr("y2", height*2) // Scaled to ensure that initial bounding box does cover graph area
                .attr("stroke", "red")
                .attr("stroke-width", 1)
                .attr("stroke-dasharray", "4,4"); // Optional: dashed line
        })
    }, [data, graphGroup, id, xScale, yScale]);

    return null;
};