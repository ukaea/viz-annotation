'use client'
import 'react-contexify/ReactContexify.css';

type GraphProps = {

    data: Array<{
        time: number,
        value: number
    }>,

    shot_id: string
}

export const DisruptionGraph = ({ data, shot_id }: GraphProps) => {
    return (
        <div>
            <h2>Disruption ({shot_id})</h2>
            <p style={{ whiteSpace: 'pre-line' }}>
                {data.map(({ time, value }) => `time: ${time}, value: ${value}`).join('\n')}
            </p>
        </div>
    )
};

