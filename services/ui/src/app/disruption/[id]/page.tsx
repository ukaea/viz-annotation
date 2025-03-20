import { DisruptionGraph } from "../components/disruputionGraph";

export default async function ShotPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  return (
    <DisruptionGraph shot_id={id} />
  );
}