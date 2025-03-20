'use client'
import 'react-contexify/ReactContexify.css';

type GraphProps = {
    shot_id: string
}

export const DisruptionGraph = ({ shot_id }: GraphProps) => {
    return (
        <p>Hello disruption ({shot_id})</p>
    )
};

