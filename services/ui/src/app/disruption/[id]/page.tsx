import { DisruptionPlot } from "../components/disruption-plot";

export default async function ShotPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const data = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/backend-api/data/disruption/${id}`)
  const json_data = await data.json()
  return (
    <DisruptionPlot data={json_data.ip} shot_id={id} />
  );
}